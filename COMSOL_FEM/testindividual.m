d=5.5673, w=28.7666, n1=3.0, R=26.14;
[ratio, Fy, G] = evaluate_individual(d, w, n1, R);
fprintf('d = %.2f mm, w = %.2f mm, n1 = %.2f, R = %.2f mm | Fy = %.4f N, G = %.4f N, Ratio = %.4f\n', ...
            d, w, n1, R, Fy, G, ratio);
