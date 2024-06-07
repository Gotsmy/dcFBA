using JuMP
using LinearAlgebra
using Ipopt
using DelimitedFiles
using JLD2
import DataFrames
import CSV
import ForwardDiff
import Dierckx

include("../00_files/postprocessing_02.jl")

function fed_batch_dFBA(N0,V0,nFE,S,t_max,t_min,V_max,c_G,glu,atp,so4,xxx,pro,vlb,vub,d)
    A_idx, B_idx, C_idx, D_idx = get_idx_lists(vlb,vub)
    nA = size(A_idx,1)
    nB = size(B_idx,1)
    nC = size(C_idx,1)
    nD = size(D_idx,1)
    println("nA = ",nA)
    println("nB = ",nB)
    println("nC = ",nC)
    println("nD = ",nD)
    
    # number of reactions
    nR = size(S,2)
    # number of metabolites
    nM = size(S,1)
    # number of tracked metabolites 
    nN = length(N0)
    # number collocation points
    nCP = 3

    println("nr of KKT variables = ",nFE*(2*nA+nB+nC+nM))
    
    #--------------------------
    # SIMULATION HYPERPARAMETERS
    
    w    = 1e-20 # weigth for flux sum minimization on the OF
    phi1 = 1e2
    phi2 = 1e1
    psi  = 1e4

    #--------------------------
    # COLLOCATION AND RADAU PARAMETERS
    colmat = [0.19681547722366  -0.06553542585020 0.02377097434822;
              0.39442431473909  0.29207341166523 -0.04154875212600;
              0.37640306270047  0.51248582618842 0.11111111111111]
    radau  = [0.15505 0.64495 1.00000]

    #--------------------------
    # JuMP MODEL SETUP
    m = Model(optimizer_with_attributes(Ipopt.Optimizer, 
            "warm_start_init_point" => "yes", 
            "print_level" => 5, 
            "linear_solver" => "ma27",
            "max_iter" => Int(1e5),
            "tol" => 1e-4, 
            "acceptable_iter" => 15, 
            "acceptable_tol" => 1e-2))
    
    #--------------------------
    # VARIABLE SET UP
    @variables(m, begin
            # Differential Equation Variables
            N[   1:nN, 1:nFE, 1:nCP]  # metabolite amounts [mmol]
            Ndot[1:nN, 1:nFE, 1:nCP]  # dN/dt [mmol/h]
            V[         1:nFE, 1:nCP]  # reactor volume [L]
            Vdot[      1:nFE, 1:nCP]  # dV/dt [L/h]
            q[   1:nR, 1:nFE       ]  # FBA fluxes [mmol/(g h)]

            # KKT Variables
            lambda_Sv[1:nM, 1:nFE]

            # A: both constrained
            alpha_ub_A[1:nA, 1:nFE]
            slack_ub_A[1:nA, 1:nFE]
            alpha_lb_A[1:nA, 1:nFE]
            slack_lb_A[1:nA, 1:nFE]

            # B: lb constrained, ub free
            alpha_lb_B[1:nB, 1:nFE]
            slack_lb_B[1:nB, 1:nFE]

            # C: lb free, ub constrained
            alpha_ub_C[1:nC, 1:nFE]
            slack_ub_C[1:nC, 1:nFE]

            # Process Variabels
            t_end       # total process length [h]
            rtFE[1:nFE] # relative length of finite elements [-]
            F[1:nFE]    # feed rate [L/h]
            S0          # initial sulfate amount [mmol]
    end)

    #--------------------------
    # GUESSING START VALUES
    
    println("Setting start values.")
    for i in 1:nFE
      for j in 1:nCP
          for k in 1:nN
              set_start_value(N[k,i,j], N0[k])
          end
          set_start_value(V[i,j],V0)
      end
    end
    for i in 1:nFE
        set_start_value(rtFE[i], 1)
    end
    println("Finished setting start values.")

    #--------------------------
    # SET UP OBJECTIVE FUNCTION AND CONSTRAINTS
    
    function get_q_P(cS,tt)
        n_CTRL_k = -5.1647228826486014e-05 *psi
        n_CTRL_d =  0.0017391744074473179  *psi
        n_delta  =  0.00013961542695773983 *psi
        qswitch  = -1
        stretch  =  2
        qP = n_CTRL_k * tt + n_CTRL_d + n_delta * (tanh((-cS-qswitch)*stretch)+1)/2
        return qP
    end
    register(m, :get_q_P, 2, get_q_P; autodiff = true)
    
    # Uptake Rates as NLexpressions
    @NLexpressions(m, begin
        # glucose
        q_G[i=1:nFE], F[i]*c_G/N[2,i,2] # L/h * mmol/L / g = mmol/(g h)
        # absolute values for length of finite elements (tFE)
        tFE[i=1:nFE], t_end/nFE*rtFE[i]
        # declining production rate
	q_P[i=1:nFE], get_q_P(N[3,i,2]/V[i,2],sum(tFE[j] for j in 1:i-1)+radau[2]*tFE[i])
    end)
    
    @NLobjective(m, Max, +
        N[4,end,end]*phi1 -
        sum(
            sum(- slack_lb_A[mc,i] for mc in 1:nA) +
            sum(- slack_ub_A[mc,i] for mc in 1:nA) +
            sum(- slack_lb_B[mc,i] for mc in 1:nB) +
            sum(- slack_ub_C[mc,i] for mc in 1:nC)
	    for i in 1:nFE)/nFE)
    
    @NLconstraints(m, begin
            # NL process constraints
            NLc_q_G[i=1:nFE], -q[glu,i] - q_G[i] == 0 # |q_G| >= |q[glu]| [mmol/(g h)]
            NLc_q_P[i=1:nFE],  q[pro,i]*psi - q_P[i] == 0
            NLc_t_end, sum(tFE[i] for i in 1:nFE) == t_end

            # NL KKT constraints, aka complementary slackness
            NLc_slack_lb_A[mc=1:nA,i=1:nFE], slack_lb_A[mc,i]  == (q[A_idx[mc],i] -vlb[A_idx[mc]])*alpha_lb_A[mc,i]/nA*phi2
            NLc_slack_ub_A[mc=1:nA,i=1:nFE], slack_ub_A[mc,i]  == (q[A_idx[mc],i] -vub[A_idx[mc]])*alpha_ub_A[mc,i]/nA*phi2
            NLc_slack_lb_B[mc=1:nB,i=1:nFE], slack_lb_B[mc,i]  == (q[B_idx[mc],i] -vlb[B_idx[mc]])*alpha_lb_B[mc,i]/nB*phi2
            NLc_slack_ub_C[mc=1:nC,i=1:nFE], slack_ub_C[mc,i]  == (q[C_idx[mc],i] -vub[C_idx[mc]])*alpha_ub_C[mc,i]/nC*phi2

            # INTEGRATION BY COLLOCATION
            # set up collocation equations - 2nd-to-nth point
            coll_N[l=1:nN, i=2:nFE, j=1:nCP], N[l,i,j] == N[l,i-1,nCP]+tFE[i]*sum(colmat[j,k]*Ndot[l,i,k] for k in 1:nCP)
            coll_V[        i=2:nFE, j=1:nCP], V[i,j]   == V[  i-1,nCP]+tFE[i]*sum(colmat[j,k]*Vdot[  i,k] for k in 1:nCP)
            # set up collocation equations - 1st point
            coll_N0[l in [1,2,4], i=[1], j=1:nCP], N[l,i,j] == N0[l] + tFE[i]*sum(colmat[j,k]*Ndot[l,i,k] for k in 1:nCP)
            coll_S0[l in [3],     i=[1], j=1:nCP], N[l,i,j] == S0    + tFE[i]*sum(colmat[j,k]*Ndot[l,i,k] for k in 1:nCP)
            coll_V0[              i=[1], j=1:nCP], V[  i,j] == V0    + tFE[i]*sum(colmat[j,k]*Vdot[  i,k] for k in 1:nCP)
    end)

    #------------------------#
    # SET UP BOUNDS
    
    for mc in 1:nR
        for i in 1:nFE
            set_lower_bound(q[mc,i],vlb[mc])
            set_upper_bound(q[mc,i],vub[mc])
            for j in 1:nCP
                for n in 1:nN
                    set_lower_bound(N[n,i,j],0)
                end
                set_lower_bound(V[  i,j],V0)
                set_upper_bound(V[  i,j],V_max)
            end
        end
    end
    set_upper_bound(S0,15)
    set_lower_bound(S0,.0)
    
    for i in 1:nFE
        set_lower_bound(rtFE[i],0.8)
        set_upper_bound(rtFE[i],1.2)
        # set_lower_bound(F[i],13.9/1000)
        # set_upper_bound(F[i],13.9/1000)
        for mc in 1:nA
            set_upper_bound(alpha_lb_A[mc,i],0)
            set_lower_bound(alpha_ub_A[mc,i],0)
        end
        for mc in 1:nB
            set_upper_bound(alpha_lb_B[mc,i],0)
        end
        for mc in 1:nC
            set_lower_bound(alpha_ub_C[mc,i],0)
        end
    end    
    set_lower_bound(t_end,t_min)
    set_upper_bound(t_end,t_max)
    
    #------------------------#
    # SET UP OTHER CONSTRAINTS
    
    @constraints(m, begin
        #------------------------#
        # DIFFERENTIAL EQUATIONS
            
        # glucose
        m1[i=1:nFE, j=1:nCP], Ndot[1,i,j] == F[i]*c_G + q[glu,i]*N[2,i,j] # mmol/h = L/h * mmol/L + mmol/(g h) * g
        # biomass
        m2[i=1:nFE, j=1:nCP], Ndot[2,i,j] == q[xxx,i]*N[2,i,j] # g/h = g/(g h) * g
        # sulfate
        m3[i=1:nFE, j=1:nCP], Ndot[3,i,j] == q[so4,i]*N[2,i,j]
        # product
        m4[i=1:nFE, j=1:nCP], Ndot[4,i,j] == q[pro,i]*N[2,i,j]
        # volume
        v1[i=1:nFE, j=1:nCP], Vdot[i,j]   == F[i]

        #------------------------#
        # SYSTEM CONSTRAINTS
            
        c_S[mc=1:nM,i=1:nFE], sum(S[mc,k] * q[k,i] for k in 1:nR) == 0
        # c_UB_S0, S0 == 30

        #------------------------#
        # KKT CONSTRAINTS
            
        Lagr_A[mc=1:nA,i=1:nFE],  d[A_idx[mc]] + w*q[A_idx[mc],i]  + alpha_lb_A[mc,i] + alpha_ub_A[mc,i] +  sum(S[k,A_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        Lagr_B[mc=1:nB,i=1:nFE],  d[B_idx[mc]] + w*q[B_idx[mc],i]  + alpha_lb_B[mc,i]                    +  sum(S[k,B_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        Lagr_C[mc=1:nC,i=1:nFE],  d[C_idx[mc]] + w*q[C_idx[mc],i]                     + alpha_ub_C[mc,i] +  sum(S[k,C_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0
        Lagr_D[mc=1:nD,i=1:nFE],  d[D_idx[mc]] + w*q[D_idx[mc],i]                                        +  sum(S[k,D_idx[mc]]*lambda_Sv[k,i] for k in 1:nM) == 0

        c_alpha_lb_A[mc=1:nA,i=1:nFE],   alpha_lb_A[mc,i]    <= 0
        c_alpha_ub_A[mc=1:nA,i=1:nFE],   alpha_ub_A[mc,i]    >= 0
        c_alpha_lb_B[mc=1:nB,i=1:nFE],   alpha_lb_B[mc,i]    <= 0
        c_alpha_ub_C[mc=1:nC,i=1:nFE],   alpha_ub_C[mc,i]    >= 0    
    end)

    #------------------------#
    # MODEL OPTIMIZATION
    
    println("Model Preprocessing Finished.")
    solveNLP = JuMP.optimize!
    status = solveNLP(m)
    t = solve_time(m)
    println("Model Finished Optimization")

    # Read out model parameters
    N_    = JuMP.value.(N[:,:,:])
    Ndot_ = JuMP.value.(Ndot[:,:,:])
    V_    = JuMP.value.(V[:,:])
    Vdot_ = JuMP.value.(Vdot[:,:])
    q_    = JuMP.value.(q[:,:])
    lambda_Sv_ = JuMP.value.(lambda_Sv[:,:])
    alpha_ub_A_ = JuMP.value.(alpha_ub_A[:,:])
    slack_ub_A_ = JuMP.value.(slack_ub_A[:,:])
    alpha_lb_A_ = JuMP.value.(alpha_lb_A[:,:])
    slack_lb_A_ = JuMP.value.(slack_lb_A[:,:])
    alpha_lb_B_ = JuMP.value.(alpha_lb_B[:,:])
    slack_lb_B_ = JuMP.value.(slack_lb_B[:,:])
    alpha_ub_C_ = JuMP.value.(alpha_ub_C[:,:])
    slack_ub_C_ = JuMP.value.(slack_ub_C[:,:])
    tFE_   = JuMP.value.(tFE[:])
    F_     = JuMP.value.(F[:])
    S0_    = JuMP.value.(S0)
    rtFE_  = JuMP.value.(rtFE[:])
    t_end_ = JuMP.value.(t_end)
    
    @JLD2.save dirname*"/variables.jld2" N_ Ndot_ V_ Vdot_ q_ lambda_Sv_ tFE_ F_ S0_ rtFE_ t_end_ alpha_ub_A_ slack_ub_A_ alpha_lb_A_ slack_lb_A_ alpha_lb_B_ slack_lb_B_ alpha_ub_C_ slack_ub_C_
    
    sum_slack = 0.0*Vector{Float64}(undef,nFE)
    for i in 1:nFE
        sum_slack[i] = sum(slack_lb_A_[j,i] for j in 1:nA; init=0) + 
                       sum(slack_ub_A_[j,i] for j in 1:nA; init=0) + 
                       sum(slack_lb_B_[j,i] for j in 1:nB; init=0) +
                       sum(slack_ub_C_[j,i] for j in 1:nC; init=0)
    end

return  N_, Ndot_, V_, Vdot_, q_, tFE_, sum_slack, m, t
end

function run_kkt_simulation()

    # load stoichiometric matrix & flux bounds
    S   = readdlm("../00_files/S.csv")
    vlb = readdlm("../00_files/LB.csv")[:,1]
    vub = readdlm("../00_files/UB.csv")[:,1]
    println("nR = ",size(vlb,1)," = ",size(vub,1))
    
    # reaction indices
    glu =  85 # glucose uptake
    atp = 321 # atp maintenance
    so4 = 330 # sulfate uptake
    xxx = 468 # biomass production
    pro = 469 # ppro production

    # open product secretion
    vlb[pro] = 0
    vub[pro] = 1000

    # open glucose uptake
    vlb[glu] = -10.5
    vub[glu] = 0

    # open biomass growth
    vlb[xxx] = 0
    vub[xxx] = 1000
    
    # initial bioprocess state variables
    V0 =  0.5  # L
    G0 =  0.0  # mmol
    X0 =  1.52 # g
    S0 =  9.55 # mmol
    N0 = [G0,X0,S0,0]
    
    # glucose concentration in feed medium
    c_G = 450/0.18015588 # g/L /(g/mmol) = mmol/L, i.el max solubility of glucose in water

    # process length and state variable bounds
    t_max = 33 # h
    t_min = 33 # h
    V_max = 0.9587 # L
    nFE   = 20

    # KKT FBA objective function: optimize product
    nR = size(S,2)
    d  = 0.0*Vector{Float64}(undef,nR) 
    d[xxx] = -1

    N_, Ndot_, V_, Vdot_, q_, tFE_, sum_slack, m_, t_ = fed_batch_dFBA(N0,V0,nFE,S,t_max,t_min,V_max,c_G,glu,atp,so4,xxx,pro,vlb,vub,d)
    println("DONE")
          
    #-------------------------
    # SAVING ALL INTERESTING FILES
    
    summary = "#-------------------------\n" * 
              "Termination Status\n" * string(termination_status(m_)) * 
              "\n\nTime Elapsed\n" * string(t_) *
              "\n\nObjective Value\n" * string(JuMP.objective_value(m_)) * 
              "\n\nNormalized Complementary Slackness\n" * string(-(sum(-sum_slack))/nFE) * 
              "\n\nFinite Elements\n" * string([round(i,digits=2) for i in tFE_]) * 
              "\n\nComplementary Slackness per Finite Element\n" * string([round(sum_slack[i],digits=3) for i in 1:nFE]) * 
              "\n\nFeed Rates\n" * string([round(i,digits=3) for i in Vdot_[:,1]]) *
              "\n\nGrowth Rates [h-1]\n" * string([round(i,digits=3) for i in q_[xxx,:]]) * 
              "\n\nFinal Product Amount [mmol]\n" * string(N_[4,end,end]) * 
              "\n\nFinal Biomass Amount [g]\n" * string(N_[2,end,end]) * 
              "\n\nS0\n" * string(value(JuMP.variable_by_name(m_,"S0"))) * 
              "\n\nProcess End\n" * string(round(sum(tFE_),digits=2)) * 
              "\n#-------------------------"
    println(summary)
    
    #-------------------------
    # POST PROCESSING

    df = DataFrames.DataFrame(hcat(
            get_time_points(tFE_),
            get_points(V_,tFE_,V0),
            get_points(N_,tFE_,[N0[1],N0[2],value(JuMP.variable_by_name(m_,"S0")),N0[4]]),
            get_points(Vdot_,tFE_),
            get_points(Ndot_,tFE_),
            get_fluxes(q_,[glu,xxx,so4,pro,atp],tFE_)),
        ["t","V","G","X","S","P","r_V","r_G","r_X","r_S","r_P","q_G","q_X","q_S","q_P","q_M"]);
    
    write(   dirname*"/summary.txt",summary)
    writedlm(dirname*"/q.csv",  q_, ',')
    writedlm(dirname*"/vlb.csv",vlb,',')
    writedlm(dirname*"/vub.csv",vub,',')
    CSV.write(dirname*"/df.csv",df) 
end

#-------------------------
# SCRIPT START

println("Start Script")
scriptname = PROGRAM_FILE
dirname = scriptname[begin:end-3]
println("Creating results directory: ",dirname)
mkpath(dirname)

#-------------------------
# RUN MODEL

debug = false
if debug == true
    run_kkt_simulation()
else
    # sometimes errors happen during initialization, they usually desappear when rerunning the simulation
    # thus in debug = false mode, these errors are escapted.
    global worked = false
    global nTRY = 0
    while worked == false
        try
            run_kkt_simulation()
            global worked = true
        catch e
            if isa(e,ErrorException)
                global nTRY += 1
                println("Error During Initialization. Retrying ... (",nTRY,")")
            else
                rethrow(e)
            end
        end
    end
end

println("Script Ended")
