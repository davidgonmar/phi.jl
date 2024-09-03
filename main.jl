using Flux
using Transformers


# ======================= RoPE =======================
function rotate_half(x::AbstractArray{Float32, 3})::AbstractArray{Float32, 3}
	x1 = x[:, 1:(end÷2), :]
	x2 = x[:, (end÷2+1):end, :]
	return cat(x1, x2, dims = 2)
end

struct RoPE
	# rotary position embedding
	freqs_cos::AbstractMatrix{Float32} # (seq_len, rotary_dim)
	freqs_sin::AbstractMatrix{Float32} # (seq_len, rotary_dim)
end
function RoPE(rotary_dim::Int, seq_len::Int)

	arange = collect((0:rotary_dim//2-1))
	thetas = 10000 .^ (-2 .* (arange) ./ rotary_dim)
	thetas = vcat(thetas, thetas)
	m = collect((0:seq_len-1))
	freqs = m * thetas' # outer product, shape (seq_len, rotary_dim)
	freqs_cos = cos.(freqs)
	freqs_sin = sin.(freqs)

	return RoPE(freqs_cos, freqs_sin)
end

function (m::RoPE)(seq::AbstractArray{Float32, 3})::AbstractArray{Float32, 3}
	sin, cos = m.freqs_sin, m.freqs_cos
	# seq (seq_len, rotary_dim, n_heads)
	# sin and cos (seq_len, rotary_dim)
	# crop sequence length
	sin, cos = sin[1:size(seq, 1), :], cos[1:size(seq, 1), :]
	sin, cos = reshape(sin, (size(sin, 1), size(sin, 2), 1)), reshape(cos, (size(cos, 1), size(cos, 2), 1))
	# print shapes
	println("sin: ", size(sin))
	println("cos: ", size(cos))
	println("seq: ", size(seq))
	seq .* cos .+ rotate_half(seq) .* sin
end
# ======================= RoPE =======================


# ======================= MLP =======================
struct MLP
	# multi-layer perceptron
	dense1::Dense
	dense2::Dense
end

function (m::MLP)(x)
	x = m.dense1(x)
	x = gelu(x)
	x = m.dense2(x)
	return x
end

function MLP(d_model::Int, d_ff::Int)
	return MLP(Dense(d_model, d_ff, gelu), Dense(d_ff, d_model, identity))
end
# ======================= MLP =======================

# ======================= SelfAttention =======================

struct SelfAttention
	# self-attention layer
	Wq::Dense
	Wk::Dense
	Wv::Dense
	Wo::Dense
	rope::RoPE
	n_heads::Int
	rotary_dim::Int
end
function (m::SelfAttention)(x::AbstractMatrix{Float32})
	d_model, seq_len = size(x)
	head_dim = Int(d_model // m.n_heads)

	q, k, v = m.Wq(x), m.Wk(x), m.Wv(x)
	new_shape = (m.n_heads, head_dim, seq_len)
	q = reshape(q, new_shape)
	k = reshape(k, new_shape)
	v = reshape(v, new_shape)

	# transpose so heads dim is the batched dim -> (seq_len, head_dim, n_heads)
	permutation = (3, 2, 1)
	q, k, v = permutedims(q, permutation), permutedims(k, permutation), permutedims(v, permutation)

	# Phi applies RoPE to tbe first rotary_dim dimensions in head_dim
	# check if rotary_dim is out of bounds
	rotary_dim = m.rotary_dim
	if rotary_dim >= size(q, 2)
		rotq = q
		origq = q[:, 1:0, :]  # empty array
	else
		rotq = q[:, 1:rotary_dim, :]
		origq = q[:, rotary_dim+1:end, :]
	end
	rotq = m.rope(rotq)
	q = cat(rotq, origq, dims = 2)

	if rotary_dim >= size(k, 2)
		rotk = k
		origk = k[:, 1:0, :]
	else
		rotk = k[:, 1:rotary_dim, :]
		origk = k[:, rotary_dim+1:end, :]
	end
	rotk = m.rope(rotk)
	k = cat(rotk, origk, dims = 2) # 



	# scaled dot-product attention
	# (seq_len, head_dim, n_heads) x (head_dim, seq_len, n_heads) -> (seq_len, seq_len, n_heads)
	att_scores = Flux.batched_mul(q, permutedims(k, (2, 1, 3))) ./ sqrt(d_model)

	# causal masking
	mask = fill(-Inf, size(att_scores, 1), size(att_scores, 2))
	mask = Flux.triu!(mask, 1)
	mask = reshape(mask, (size(mask, 1), size(mask, 2), 1)) # broadcast to n_heads (batch dim)
	att_scores = softmax(att_scores .+ mask, dims = 2) # (seq_len, seq_len, n_heads)

	# gather values (seq_len, seq_len, n_heads) x (seq_len, head_dim, n_heads) -> (seq_len, head_dim, n_heads)
	vals = Flux.batched_mul(att_scores, v) # (seq_len, head_dim, n_heads)

	# back to original shape (d_model, seq_len)
	vals = permutedims(vals, (3, 2, 1)) # (n_heads, head_dim, seq_len)
	vals = reshape(vals, (d_model, seq_len))
	return m.Wo(vals)
end

function SelfAttention(d_model::Int, n_heads::Int, rotary_dim::Int)
	return SelfAttention(Dense(d_model, d_model, identity), Dense(d_model, d_model, identity), Dense(d_model, d_model, identity), Dense(d_model, d_model, identity), RoPE(rotary_dim, 512), n_heads, rotary_dim)
end
# ======================= SelfAttention =======================

# ======================= Decoder =======================
struct Decoder
	# decoder layer
	self_attn::SelfAttention
	mlp::MLP
	ln::LayerNorm
end

function (m::Decoder)(x::AbstractMatrix{Float32})
	# phi does layer normalization pre-attention
	# x of shape (d_model, seq_len)
	x = m.ln(x)

	attn_res = m.self_attn(x)
	mlp_res = m.mlp(x)
	return x + attn_res + mlp_res
end
# ======================= Decoder =======================

# ======================= Transformer =======================
struct Transformer
	# transformer model
	decoders::Vector{Decoder}
	embeddings::Embedding
	ln::LayerNorm
	dense::Dense
end

function (m::Transformer)(x::AbstractVector{Int})
	println("start embedding")
	println(size(x))
	x = m.embeddings(x)
	println(size(x))
	println("finished embedding")
	i = 1
	for (i, decoder) in enumerate(m.decoders)
		println("decoder ", i + 1)
		x = decoder(x)
	end

	x = m.ln(x)
	x = m.dense(x)
	return x
end
# ======================= Transformer =======================



# ======================= Main =======================
function main()
	# load the phi tokenizer
	weights = Transformers.load_state_dict("microsoft/phi-1")
	cfg = Transformers.load_config("microsoft/phi-1")
	#n_hidden_layers = cfg["n_hidden_layers"]
	println("cfg: ", cfg)
	vocab_size = cfg["vocab_size"]
	num_hidden_layers = cfg["num_hidden_layers"]
	d_ffn = cfg["intermediate_size"]
	num_attention_heads = cfg["num_attention_heads"]
	d_model = cfg["hidden_size"]

	partial_rotary_factor = cfg["partial_rotary_factor"]

	rotary_dim = Int((d_model // num_attention_heads) * partial_rotary_factor)

	for (key, value) in weights
		println("name: ", key, " shape: ", size(value))
	end
	tokenizer = Transformers.load_tokenizer("microsoft/phi-1")
	input = Transformers.encode(tokenizer, "Hello, world!").token
	input = Flux.onecold(input, 1:size(input, 1))

	println("input: ", input)

	# model
	transformer =
		Transformer([Decoder(SelfAttention(d_model, num_attention_heads, rotary_dim), MLP(d_model, d_ffn), LayerNorm(d_model)) for _ in 1:num_hidden_layers], Embedding(vocab_size, d_model), LayerNorm(d_model), Dense(d_model, vocab_size, identity))


	# forward
	y = transformer(input)

	# shape (vocab_size, seq_len)


	# decode. y are probabilities of shape (seq_len, vocab_size)

	# get the most likely token and materialize it
	token = Flux.onecold(y, 1:size(y, 1))


	# decode token
	token = Transformers.decode(tokenizer, token)

	println("output: ", token)


end

main()
