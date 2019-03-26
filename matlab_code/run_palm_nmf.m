load('output.mat')

params.init_W = init_W;
params.init_H = init_H;
params.max_iter = 1;

[W,H,objective_function,iteration_times] = palm_nmf(V,params);



objective_function
norm(V-init_W*init_H,'fro')
params.r = 5
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
objective_function
norm(W,'fro')
norm(H,'fro')
params.betaH = 0;
params.betaW = 0;
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
norm(H,'fro')
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
params.betaH = 0;
params.betaW = 0;
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
norm(H,'fro')
norm(W,'fro')
objective_function
sprintf('%.9f',objective_function)
params.maxiter = 200;
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
params.max_iter = 200;
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
objective_function
objective_function(end)
sprintf('%.9f',objective_function(end))
diag(ones(5-1,1),-1);
A = diag(ones(5-1,1),-1);
A
params.sparsity = 1;
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
load('output.mat')
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
params.sparsity = 1;
params.r = 5
params.max_iter = 200;
params.max_iter = 1;
params.betaW = 0;
params.betaH = 0;
params.init_W = init_W;
params.init_H = init_H;
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
objective_function(end)
sprintf('%.9f',objective_function(end))
params
params.smoothness = 1
params.sparsity = 0
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
sprintf('%.9f',objective_function(end))
params.sparsity = 1
[W,H,objective_function,iteration_times] = palm_nmf(V,params);
sprintf('%.9f',objective_function(end))









