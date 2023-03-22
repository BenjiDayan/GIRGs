
function get_dists(pts::Matrix{Float64}, torus_side_length::Float64)
	n = size(pts)[1]
	lpts = repeat(pts, n)
	rpts::Matrix{Float64} = repeat(pts, inner=[n, 1])
	diff = abs.(rpts - lpts)
	torus_diff = min.(diff, torus_side_length .- diff)
	dists = maximum(torus_diff, dims=2)
	return reshape(dists, (n, n))
end


# function get_edges(pts::Matrix, weights, alpha, torus_side_length)
# 	n = size(pts)[1]
#     d = size(pts)[2]
#     edges = zeros(UInt8, (n, n))
#     for i in 1:n
#         diff = abs.(3)
#     end
# end


# # This might be an approach that returns an adjacency matrix eventually and never has a
# # nxn float16 matrix. However we'd need a better temporary storage
# function get_diststemp(pts::Matrix, torus_side_length)
# 	n = size(pts)[1]
#     d = size(pts)[2]
#     output = zeros(Float16, n, n)
#     for i in 1:n
#         i_to_other_diff = abs.(reshape(pts[i, :], (1, 1, d)) .- reshape(pts[i+1:end, :], (1, n-i, d)))[1, :, :]
#         i_to_other_torus_diff = min.(i_to_other_diff, torus_side_length .- i_to_other_diff)
#         i_to_other_dists = maximum(i_to_other_torus_diff, dims=2)
#         output[i, i+1:end] = i_to_other_dists
#         output[i+1:end, i] = i_to_other_dists
#     end
#     return output
# end

function get_dists_novars(pts::Matrix, torus_side_length)
    torus_side_length = convert(eltype(pts), torus_side_length)
	n = size(pts)[1]
    d = size(pts)[2]
	diff = abs.(reshape(pts, (n, 1, d)) .- reshape(pts, (1, n, d)))
	torus_diff = min.(diff, torus_side_length .- diff)
	dists = maximum(torus_diff, dims=3)
	return reshape(dists, (n, n))
end


# julia> n = 10000; d = 3; @time get_dists2(convert(Matrix{Float16}, rand(n, d)),Float16[n^(1/d)]);
#   1.064116 seconds (32 allocations: 1.304 GiB, 4.62% gc time)

# julia> n = 10000; d = 3; @time get_dists2(rand(n, d), n^(1/d));
#   3.149429 seconds (29 allocations: 5.216 GiB, 1.55% gc time)


function get_dists_itty(pts::Matrix{Float64}, torus_side_length::Float64)
	n = size(pts)[1]
    output = zeros(Float64, n, n)
    for i in 1:n
        for j in i+1:n
            diff = abs.(pts[i, :] - pts[j, :])
            torus_diff = min.(diff, torus_side_length .- diff)
            val = maximum(torus_diff)
            output[i, j] = val
            output[j, i] = val
        end
    end
    return output
end




# n = 5000
# d = 3
# pts = rand(n, d)
# tsl = n^(1/d)
# @profview dists = get_dists(pts, tsl)
# println("done")
