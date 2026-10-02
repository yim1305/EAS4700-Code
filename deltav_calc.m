%% Lunar Hopper - Lambert Delta-V Optimization
clear; clc;

%% Mission
lat1 = -85.5;          % Launch latitude [deg]
lon1 = -87.1;          % Launch longitude [deg]

lat2 = -78.9044;       % Landing latitude [deg]
lon2 = -87.1;          % Landing longitude [deg]

TOF_search_start = 100;     % Start looking for valid Lambert solution [s]
TOF_max = 1200;             % Maximum allowable TOF [s]
searchStep = 5;             % Used only to find feasible lower bound [s]

%% Moon Constants
mu = 4.9048695e12;           % Lunar gravitational parameter [m^3/s^2]
Rmoon = 1737.4e3;            % Mean lunar radius [m]

Tmoon = 27.321661*86400;     % Sidereal rotation period [s]
omega = 2*pi/Tmoon;          % Lunar rotation rate [rad/s]
omegaVec = [0; 0; omega];

%% Moon-Fixed Surface Positions
r1_fixed = latlon2cart(lat1,lon1,Rmoon);
r2_fixed = latlon2cart(lat2,lon2,Rmoon);

thetaSurface = acos(dot(r1_fixed,r2_fixed) / ...
    (norm(r1_fixed)*norm(r2_fixed)));

surfaceDistance = Rmoon*thetaSurface;

fprintf('Surface Distance = %.2f km\n',surfaceDistance/1000);

%% Find First Feasible TOF
TOF_test = TOF_search_start:searchStep:TOF_max;

TOF_min = NaN;

for TOF = TOF_test

    J = hopCost(TOF,r1_fixed,r2_fixed,mu,omega,omegaVec);

    if isfinite(J)
        TOF_min = TOF;
        break
    end

end

if isnan(TOF_min)
    error('No feasible Lambert solution found.');
end

fprintf('First Feasible TOF = %.1f s\n',TOF_min);

%% Continuous Delta-V Optimization
costFun = @(TOF) hopCost(TOF, ...
    r1_fixed,r2_fixed,mu,omega,omegaVec);

options = optimset('TolX',1e-6);

[TOF_opt,DV_opt] = fminbnd( ...
    costFun,TOF_min,TOF_max,options);

%% Get Details at Optimal TOF
[~,DV_launch_opt,DV_land_opt,a_opt,v1_opt,v2_opt, ...
    DV_launch_vec_opt,DV_land_vec_opt] = ...
    hopCost(TOF_opt,r1_fixed,r2_fixed,mu,omega,omegaVec);

fprintf('\n===== OPTIMAL LAMBERT TRAJECTORY =====\n');
fprintf('Optimal TOF       = %.3f s\n',TOF_opt);
fprintf('                  = %.3f min\n',TOF_opt/60);

fprintf('Semi-major Axis   = %.3f km\n',a_opt/1000);

fprintf('Launch Delta-V    = %.3f m/s\n',DV_launch_opt);
fprintf('Landing Delta-V   = %.3f m/s\n',DV_land_opt);

fprintf('Launch Delta-V Vector  = [%.3f %.3f %.3f] m/s\n', ...
    DV_launch_vec_opt(1),DV_launch_vec_opt(2),DV_launch_vec_opt(3));

fprintf('Landing Delta-V Vector = [%.3f %.3f %.3f] m/s\n', ...
    DV_land_vec_opt(1),DV_land_vec_opt(2),DV_land_vec_opt(3));

fprintf('Total Hop Delta-V = %.3f m/s\n',DV_opt);
fprintf('                  = %.4f km/s\n',DV_opt/1000);

%% Plot Cost Function
TOF_plot = linspace(TOF_min,TOF_max,500);
DV_plot = nan(size(TOF_plot));

for k = 1:length(TOF_plot)

    DV_plot(k) = hopCost( ...
        TOF_plot(k), ...
        r1_fixed,r2_fixed, ...
        mu,omega,omegaVec);

end

valid = isfinite(DV_plot);

figure

plot(TOF_plot(valid),DV_plot(valid), ...
    'LineWidth',1.5)

hold on

plot(TOF_opt,DV_opt,'o', ...
    'MarkerSize',9, ...
    'MarkerFaceColor','auto')

grid on

xlabel('Time of Flight [s]')
ylabel('One-Way \DeltaV [m/s]')
title('Lunar Hopper \DeltaV Optimization')

legend('Lambert Cost Function', ...
       'Minimum \DeltaV', ...
       'Location','best')

xlim([TOF_min TOF_max])


%% ================= FUNCTIONS =================

function [J,DV_launch,DV_land,a,v1,v2,DV_launch_vec,DV_land_vec] = ...
    hopCost(TOF,r1_fixed,r2_fixed,mu,omega,omegaVec)

    J = Inf;
    DV_launch = NaN;
    DV_land = NaN;
    a = NaN;
    v1 = [NaN;NaN;NaN];
    v2 = [NaN;NaN;NaN];
    DV_launch_vec = [NaN;NaN;NaN];
    DV_land_vec = [NaN;NaN;NaN];

    %% Rotate Landing Site During Flight
    r1 = r1_fixed;

    Rrot = rotz3(omega*TOF);
    r2 = Rrot*r2_fixed;

    vSite1 = cross(omegaVec,r1);
    vSite2 = cross(omegaVec,r2);

    %% Lambert Geometry
    r1mag = norm(r1);
    r2mag = norm(r2);

    c = norm(r2-r1);

    s = (r1mag+r2mag+c)/2;

    amin = s/2;

    %% Minimum-Energy Transfer Time
    alphaME = pi;

    betaME = 2*asin( ...
        sqrt((s-c)/(2*amin)) );

    tME = sqrt(amin^3/mu) * ...
        ((alphaME-sin(alphaME)) - ...
         (betaME-sin(betaME)));

    %% Find Semi-Major Axis for Requested TOF
    TOFfun = @(a) ...
        lambertTOF(a,s,c,mu,TOF,tME);

    aLow = amin*(1+1e-8);
    aHigh = 2*amin;

    FLow = TOFfun(aLow);
    FHigh = TOFfun(aHigh);

    while sign(FLow) == sign(FHigh)

        aHigh = 2*aHigh;

        if aHigh > 1e12
            return
        end

        FHigh = TOFfun(aHigh);

    end

    try
        a = fzero(TOFfun,[aLow aHigh]);
    catch
        return
    end

    %% Alpha and Beta
    alpha0 = 2*asin( ...
        sqrt(s/(2*a)) );

    beta = 2*asin( ...
        sqrt((s-c)/(2*a)) );

    if TOF <= tME
        alpha = alpha0;
    else
        alpha = 2*pi-alpha0;
    end

    deltaE = alpha-beta;

    %% Lagrange f and g
    f = 1 - (a/r1mag)*(1-cos(deltaE));

    g = TOF - sqrt(a^3/mu)* ...
        (deltaE-sin(deltaE));

    gdot = 1 - (a/r2mag)*(1-cos(deltaE));

    %% Transfer Velocities
    v1 = (r2-f*r1)/g;

    v2 = (gdot*r2-r1)/g;

    %% Delta-V Cost
    DV_launch_vec = v1-vSite1;

    DV_land_vec = vSite2-v2;

    DV_launch = norm(DV_launch_vec);

    DV_land = norm(DV_land_vec);

    J = DV_launch + DV_land;

end


function F = lambertTOF(a,s,c,mu,TOF,tME)

    alpha0 = 2*asin( ...
        sqrt(s/(2*a)) );

    beta = 2*asin( ...
        sqrt((s-c)/(2*a)) );

    if TOF <= tME
        alpha = alpha0;
    else
        alpha = 2*pi-alpha0;
    end

    tCalc = sqrt(a^3/mu) * ...
        ((alpha-sin(alpha)) - ...
         (beta-sin(beta)));

    F = tCalc-TOF;

end


function r = latlon2cart(latDeg,lonDeg,R)

    lat = deg2rad(latDeg);
    lon = deg2rad(lonDeg);

    r = R * [ ...
        cos(lat)*cos(lon);
        cos(lat)*sin(lon);
        sin(lat)];

end


function R = rotz3(theta)

    R = [cos(theta) -sin(theta) 0;
         sin(theta)  cos(theta) 0;
         0           0          1];

end