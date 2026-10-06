#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

// Implements equations (3)--(5) of pss_convergence_proof-3.pdf.
extern "C" int pss_estimate(const double* x, int n, int d, int ell, double* out) {
    try {
        std::fill(out, out + 12, 0.0);
        if (n < 2 || d < 1 || ell < 1) return 1;
        std::vector<double> lo(d), hi(d);
        for (int j = 0; j < d; ++j) {
            lo[j] = hi[j] = x[j];
            for (int i = 0; i < n; ++i) {
                double v = x[i*d+j];
                if (!std::isfinite(v)) return 2;
                lo[j] = std::min(lo[j], v);
                hi[j] = std::max(hi[j], v);
            }
            if (hi[j] <= lo[j]) return 3;
        }
        std::vector<std::pair<std::uint64_t, int>> rows(n);
        for (int i = 0; i < n; ++i) {
            std::uint64_t key = 0;
            for (int j = 0; j < d; ++j) {
                int b = static_cast<int>(std::ceil(ell*(x[i*d+j]-lo[j])/(hi[j]-lo[j]))) - 1;
                b = std::max(0, std::min(ell-1, b));
                key = key*ell + b;
            }
            rows[i] = {key, i};
        }
        std::sort(rows.begin(), rows.end());
        double log_sum = 0, legacy_log_sum = 0, integrated_mass = 0;
        int valid_count = 0, occupied = 0, singletons = 0, min_size = n;
        int boundary_valid = 0, small_cell_points = 0;
        for (int start = 0; start < n;) {
            int end = start + 1;
            while (end < n && rows[end].first == rows[start].first) ++end;
            int s = end-start;
            ++occupied;
            min_size = std::min(min_size, s);
            if (s < 2) { ++singletons; ++small_cell_points; start=end; continue; }
            int m = static_cast<int>(std::floor(std::sqrt(s)+0.5));
            if (s < 2*m+1) small_cell_points += s;
            double log_weight = std::log(static_cast<double>(s)/n);
            std::vector<double> logf(s, log_weight);
            std::vector<unsigned char> valid(s, 1), boundary(s, 0);
            double cell_mass = static_cast<double>(s)/n;
            legacy_log_sum += s*log_weight;
            for (int j = 0; j < d; ++j) {
                std::vector<std::pair<double,int>> sorted(s);
                for (int i=0; i<s; ++i) sorted[i] = {x[rows[start+i].second*d+j],i};
                std::sort(sorted.begin(), sorted.end());
                auto T = [&](int r) { return sorted[std::max(1,std::min(s,r))-1].first; };
                std::vector<double> xi(s+2), delta(s+1);
                xi[0] = T(1); xi[s+1] = T(s);
                double window = 0;
                for (int u=1-m; u<=m; ++u) window += T(u);
                for (int r=1; r<=s; ++r) {
                    if (r>1) window += T(r+m-1)-T(r-m-1);
                    xi[r] = std::max(xi[r-1], std::min(T(s), window/(2*m)));
                }
                double margin_mass = 0;
                for (int r=0; r<=s; ++r) {
                    delta[r] = T(r+m)-T(r-m);
                    if (delta[r]>0) margin_mass += (xi[r+1]-xi[r])*(2.0*m)/(s*delta[r]);
                }
                cell_mass *= margin_mass;
                for (int r=1; r<=s; ++r) {
                    double gap = T(r+m)-T(r-m);
                    legacy_log_sum += std::log(2.0*m)-std::log(static_cast<double>(s))-std::log(gap);
                    int a = static_cast<int>(std::lower_bound(xi.begin(),xi.end(),T(r))-xi.begin())-1;
                    a = std::max(0,std::min(s,a));
                    int orig = sorted[r-1].second;
                    if (!(delta[a]>0)) valid[orig] = 0;
                    else logf[orig] += std::log(2.0*m)-std::log(static_cast<double>(s))-std::log(delta[a]);
                    if (a-m<1 || a+m>s) boundary[orig]=1;
                }
            }
            integrated_mass += cell_mass;
            for (int i=0; i<s; ++i) if (valid[i]) {
                log_sum += logf[i]; ++valid_count; boundary_valid += boundary[i];
            }
            start=end;
        }
        out[0] = valid_count ? -log_sum/valid_count : 0.0;
        out[1] = -legacy_log_sum/n;
        out[2] = static_cast<double>(valid_count)/n;
        out[3] = occupied;
        out[4] = min_size;
        out[5] = static_cast<double>(n)/occupied;
        out[6] = singletons;
        out[7] = static_cast<double>(boundary_valid)/n;
        out[8] = integrated_mass;
        out[9] = static_cast<double>(small_cell_points)/n;
        out[10] = valid_count;
        out[11] = std::pow(static_cast<double>(ell),d);
        return 0;
    } catch (...) { return 4; }
}
