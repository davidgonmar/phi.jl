using Flux
#using Transformers


# ======================= RoPE =======================
function rotate_half(x::AbstractArray{Float32, 3})::AbstractArray{Float32, 3}
	x1 = x[:, :, 1:(end÷2)]
	x2 = x[:, :, (end÷2+1):end]
	return cat(x1, x2, dims = 3)
end

struct RoPE
	# rotary position embedding
	freqs_cos::AbstractMatrix{Float32} # (seq_len, d_model//2)
	freqs_sin::AbstractMatrix{Float32}
end
function RoPE(d_model::Int, seq_len::Int)

	arange = collect((0:d_model//2))
	thetas = 10000 .^ (-2 .* (arange .- 1) ./ d_model)
	thetas = vcat(thetas, thetas)
	m = collect((0:seq_len-1))
	freqs = m * thetas' # outer product
	freqs_cos = cos.(freqs)
	freqs_sin = sin.(freqs)

	return RoPE(freqs_cos, freqs_sin)
end

function (m::RoPE)(seq::AbstractArray{Float32, 3}, cos::AbstractMatrix{Float32}, sin::AbstractMatrix{Float32})::AbstractArray{Float32, 3}
	sin, cos = reshape(sin, (1, size(sin, 1), size(sin, 2))), reshape(cos, (1, size(cos, 1), size(cos, 2)))
	# crop
	sin, cos = sin[:, :, 1:size(seq, 3)], cos[:, :, 1:size(seq, 3)]
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
function bmm_3d(A, B)

	return permutedims(Flux.batched_mul(permutedims(A, (2, 3, 1)), permutedims(B, (2, 3, 1))), (3, 1, 2))
end


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
	seq_len, d_model = size(x)
	head_dim = Int(d_model // m.n_heads)

	q, k, v = m.Wq(x), m.Wk(x), m.Wv(x)
	q = reshape(q, (seq_len, m.n_heads, head_dim))
	k = reshape(k, (seq_len, m.n_heads, head_dim))
	v = reshape(v, (seq_len, m.n_heads, head_dim))

	# transpose so heads dim is first
	q, k, v = permutedims(q, (2, 1, 3)), permutedims(k, (2, 1, 3)), permutedims(v, (2, 1, 3))

	# q and k are now (seq_len, n_heads, d_model)

	# Phi applies RoPE to tbe first rotary_dim dimensions
	# check if rotary_dim is out of bounds
	rotary_dim = m.rotary_dim
	if rotary_dim >= size(q, 3)
		rotq = q
		origq = q[:, :, 1:0]  # empty array
	else
		rotq = q[:, :, 1:rotary_dim]
		origq = q[:, :, rotary_dim+1:end]
	end
	rotq = m.rope(rotq, m.rope.freqs_cos, m.rope.freqs_sin)
	q = cat(rotq, origq, dims = 3)

	if rotary_dim >= size(k, 3)
		rotk = k
		origk = k[:, :, 1:0]
	else
		rotk = k[:, :, 1:rotary_dim]
		origk = k[:, :, rotary_dim+1:end]
	end
	rotk = m.rope(rotk, m.rope.freqs_cos, m.rope.freqs_sin)
	k = cat(rotk, origk, dims = 3)



	# scaled dot-product attention
	att_scores = (bmm_3d(q, permutedims(k, (1, 3, 2)))) ./ sqrt(head_dim)

	# causal masking
	mask = fill(-Inf, size(att_scores, 2), size(att_scores, 3))
	mask = Flux.triu!(mask, 1)
	mask = reshape(mask, (1, size(mask, 1), size(mask, 2)))
	att_scores = softmax(att_scores .+ mask, dims = 3)


	# transpose back to (seq_len, n_heads, head_dim)
	att = permutedims(bmm_3d(att_scores, v), (2, 1, 3))

	# concat heads
	att = reshape(att, (seq_len, d_model))
	return m.Wo(att)
end

function SelfAttention(d_model::Int, n_heads::Int, rotary_dim::Int)
	return SelfAttention(Dense(d_model, d_model, identity), Dense(d_model, d_model, identity), Dense(d_model, d_model, identity), Dense(d_model, d_model, identity), RoPE(d_model, 512), n_heads, rotary_dim)
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
	x = m.embeddings(x)
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
	# model
	transformer = Transformer([Decoder(SelfAttention(512, 8, 128), MLP(512, 2048), LayerNorm(512)) for _ in 1:6], Embedding(10000, 512), LayerNorm(512), Dense(512, 10000, identity))

	# input
	x = rand(1:10000, 512)

	# forward
	y = transformer(x)
end

main()
