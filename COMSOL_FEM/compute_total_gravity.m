function G_total = compute_total_gravity(d, w, d1, n1, R, rho, rho2)
%COMPUTE_TOTAL_GRAVITY Total gravity of sector magnets + circular ring (N).
%
%   G_total = COMPUTE_TOTAL_GRAVITY(d, w, d1, n1, R, rho, rho2) computes
%   the total gravitational force of a wheel composed of 16 sector magnets
%   and a circular support ring.
%
%   Inputs:
%       d    - magnet radial thickness (mm)
%       w    - axial height (mm)
%       d1   - ring radial thickness (mm)
%       n1   - arc-length ratio (kept for API compatibility, not used)
%       R    - outer radius (mm)
%       rho  - magnet density (g/mm^3)
%       rho2 - ring density (g/mm^3)
%
%   Output:
%       G_total - total gravity (N)

    % Number of magnets
    n = 16;

    % ---------- 1. Sector magnets ----------
    theta_deg = 360 / n - 1;                 % sector angle (deg)
    area_magnet = (theta_deg / 360) * pi * (R^2 - (R - d)^2);
    volume_magnet = area_magnet * w;
    mass_single = volume_magnet * rho;
    mass_total_magnet = n * mass_single;

    % ---------- 2. Circular ring ----------
    R_outer = R - d - 0.5;
    R_inner = R_outer - d1;
    area_ring = pi * (R_outer^2 - R_inner^2);
    volume_ring = area_ring * w;
    mass_ring = volume_ring * rho2;

    % ---------- Total mass and gravity ----------
    mass_total = mass_total_magnet + mass_ring;      % g
    G_total = mass_total * 9.80665 / 1000;           % N
end
