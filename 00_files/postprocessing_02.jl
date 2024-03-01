function expand(M)
    return reshape(M,(1,size(M)...))
end

function get_nTP(tCI_)
    nCP = 3
    nCI = length(tCI_)
    nTP = nCI*nCP+1
    return nCP, nCI, nTP
end

function get_time_points(tCI_)
    nCP, nCI, nTP = get_nTP(tCI_)
    radau  = [0.15505 0.64495 1.00000]
    
    # initialize vector
    t = Vector{Float64}(undef,nTP)
    # initial value
    t[1] = 0
    # fill values
    for i in 1:length(tCI_)
        t[i*3-1:i*3+1] .= sum(tCI_[begin:i-1]) .+ vec(radau) * tCI_[i]
    end
    return t
end

function get_points(N_,tCI_,N0=false)
    nCP, nCI, nTP = get_nTP(tCI_)
    
    # if necessary, expand matrix
    if length(size(N_)) == 2
        N_ = expand(N_)
    end
    
    nN = size(N_,1)
    
    # create NaN initial points if they do not exist
    if N0 == false
        N0 = Vector{Float64}(undef,nN)
        N0 .= NaN
    end

    # initialize matrix
    N  = Matrix{Float64}(undef,nN,nTP)
    # fill initial values
    N[:,1] .= N0
    for i in 1:nN
        N[i,2:end] = vec(transpose(N_[i,:,:]))
    end
    return transpose(N)
end

function get_fluxes(q_,idx,tCI_)
    # extract fluxes from q_[[idx],:] and expand to fit time points
    tmp = []
    for i in idx
       tmp = vcat(tmp,expand(hcat(q_[i,:],q_[i,:],q_[i,:])))
    end
    q   = get_points(tmp,tCI_)
    return q
end

function get_idx_lists(vlb,vub)
    A_list = Int[] # both constrained (c)
    B_list = Int[] # lb c, ub uc
    C_list = Int[] # lb uc, ub c
    D_list = Int[] # both uc

    for i in 1:size(vlb,1)
        tlb = vlb[i]
        tub = vub[i]
        if isapprox(-1000,tlb,atol=1e-8) & isapprox(1000,tub,atol=1e-8)
            push!(D_list,i)
        elseif isapprox(-1000,tlb,atol=1e-8)
            push!(C_list,i)
        elseif isapprox(1000,tub,atol=1e-8)
            push!(B_list,i)
        else
            push!(A_list,i)
        end
    end
    return A_list, B_list, C_list, D_list
end

function fake_get_idx_lists(vlb,vub)
    # this function returns the same list as get_idx_lists, but all lists contain all indices, 
    # i.e., no KKT constraint reduction is done
    A_list = Int[] # both constrained (c)
    B_list = Int[] # lb c, ub uc
    C_list = Int[] # lb uc, ub c
    D_list = Int[] # both uc

    for i in 1:size(vlb,1)
        push!(A_list,i)
    end
    return A_list, B_list, C_list, D_list	
end
