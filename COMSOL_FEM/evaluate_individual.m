function [ratio, Fy, G] = evaluate_individual(d, w, n1, R)
%EVALUATE_INDIVIDUAL Evaluate one wheel design using COMSOL and Fy/G ratio.
%
%   [ratio, Fy, G] = EVALUATE_INDIVIDUAL(d, w, n1, R) runs the COMSOL model
%   for a given set of geometric parameters and returns the vertical
%   magnetic force Fy, the total gravity G, and their ratio Fy/G.
%
%   Inputs:
%       d   - magnet radial thickness (mm)
%       w   - axial height (mm)
%       n1  - arc-length ratio (dimensionless)
%       R   - outer radius (mm)
%
%   Outputs:
%       ratio - Fy / G (dimensionless)
%       Fy    - vertical magnetic force (N)
%       G     - total gravity (N)

    import com.comsol.model.*
    import com.comsol.model.util.*
    ModelUtil.showProgress(true);

    rho  = 7.5e-3;  % magnet density (g/mm^3)
    rho2 = 7.8e-3;  % yoke density (g/mm^3)
    d1   = 1.0;     % fixed yoke thickness (mm)

    try
        % Open COMSOL model
        model = mphopen('model\wheel.mph');

        % Set parameters
        model.param.set('d',  sprintf('%.3f[mm]', d));
        model.param.set('w',  sprintf('%.3f[mm]', w));
        model.param.set('d1', '1[mm]');
        model.param.set('n1', sprintf('%.3f',   n1));
        model.param.set('R',  sprintf('%.3f[mm]', R));

        % Update geometry, mesh, and solve
        model.geom('geom1').runAll;
        model.mesh('mesh1').run;
        model.study('std1').feature('stat').set('initmethod', 'init');
        model.study('std1').run;

        % Extract simulation result
        Fy = mphglobal(model, 'mfnc.Forcey_0');

        % Compute gravity
        G = compute_total_gravity(d, w, d1, n1, R, rho, rho2);

        % Compute ratio
        ratio = Fy / G;

        % Log result
        fprintf(['d = %.2f mm, w = %.2f mm, n1 = %.2f, R = %.2f mm | ', ...
                 'Fy = %.4f N, G = %.4f N, Ratio = %.4f\n'], ...
                d, w, n1, R, Fy, G, ratio);

        % Close model
        ModelUtil.remove(model.tag());

    catch
        warning(['COMSOL simulation failed for d = %.2f, w = %.2f, ', ...
                 'n1 = %.2f, R = %.2f; returning a small fallback value.'], ...
                 d, w, n1, R);
        Fy    = 1e-6;
        G     = 1;
        ratio = Fy / G;
    end
end
