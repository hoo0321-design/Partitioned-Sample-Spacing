#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <vector>

struct Margin {
    std::vector<double> xi, log_density;
};
struct Cell {
    std::vector<int> rows;
    std::vector<Margin> margins;
};

// The same fitted sub-grid evaluates both training and held-out observations.
extern "C" int pss_v2_evaluate(const double* x, int n, int d, int ell,
                               const double* q, int nq, double* logf,
                               int* sizes, int* reasons, double* summary) {
    try {
        const double neginf = -std::numeric_limits<double>::infinity();
        std::fill(logf, logf+nq, neginf);
        std::fill(sizes, sizes+nq, 0);
        std::fill(reasons, reasons+nq, 6);
        std::fill(summary, summary+5, 0.0);
        if (n < 1 || d < 1 || ell < 1 || nq < 0) return 1;
        std::uint64_t total = 1;
        for (int j=0; j<d; ++j) {
            if (total > static_cast<std::uint64_t>(INT64_MAX)/ell) return 1;
            total *= ell;
        }
        std::vector<double> lo(d), hi(d);
        bool degenerate = n < 2;
        for (int j=0; j<d; ++j) {
            std::vector<double> sorted(n);
            for (int i=0; i<n; ++i) {
                if (!std::isfinite(x[i*d+j])) return 1;
                sorted[i] = x[i*d+j];
            }
            std::sort(sorted.begin(), sorted.end());
            lo[j] = sorted.front(); hi[j] = sorted.back();
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
                degenerate = true;
        }
        for (int i=0; i<nq*d; ++i) if (!std::isfinite(q[i])) return 1;
        if (degenerate) { summary[4]=1; return 0; }
        auto key_of = [&](const double* row) {
            std::uint64_t key=0;
            for (int j=0; j<d; ++j) {
                // Left-open/right-closed, including the observed minimum.
                int b=static_cast<int>(std::ceil(ell*(row[j]-lo[j])/(hi[j]-lo[j])))-1;
                key=key*ell+std::max(0,std::min(ell-1,b));
            }
            return key;
        };
        std::map<std::uint64_t,Cell> cells;
        for (int i=0; i<n; ++i) cells[key_of(x+i*d)].rows.push_back(i);
        summary[0]=cells.size(); summary[1]=n;
        for (auto& item: cells) {
            Cell& cell=item.second;
            int s=cell.rows.size();
            summary[1]=std::min(summary[1],static_cast<double>(s));
            if (s<2) { summary[2]+=1; continue; }
            int m=static_cast<int>(std::floor(std::sqrt(s)+0.5));
            double cell_mass=static_cast<double>(s)/n;
            for (int j=0; j<d; ++j) {
                std::vector<double> sorted(s);
                for (int i=0; i<s; ++i) sorted[i]=x[cell.rows[i]*d+j];
                std::sort(sorted.begin(), sorted.end());
                auto T=[&](int r) { return sorted[std::max(1,std::min(s,r))-1]; };
                Margin margin;
                margin.xi.resize(s+2); margin.log_density.resize(s+1,neginf);
                margin.xi[0]=T(1); margin.xi[s+1]=T(s);
                long double window=0;
                for (int u=1-m; u<=m; ++u) window+=T(u);
                for (int r=1; r<=s; ++r) {
                    if (r>1) window+=static_cast<long double>(T(r+m-1))-T(r-m-1);
                    margin.xi[r]=std::max(margin.xi[r-1],std::min(T(s),static_cast<double>(window/(2*m))));
                }
                double mass=0;
                for (int r=0; r<=s; ++r) {
                    double gap=T(r+m)-T(r-m);
                    if (gap>0) {
                        margin.log_density[r]=std::log(2.0*m)-std::log(s)-std::log(gap);
                        mass+=(margin.xi[r+1]-margin.xi[r])*std::exp(margin.log_density[r]);
                    }
                }
                cell_mass*=mass;
                cell.margins.push_back(std::move(margin));
            }
            summary[3]+=cell_mass;
        }
        for (int i=0; i<nq; ++i) {
            const double* row=q+i*d;
            bool outside=false;
            for (int j=0; j<d; ++j) outside |= row[j]<lo[j] || row[j]>hi[j];
            if (outside) { reasons[i]=1; continue; }
            auto found=cells.find(key_of(row));
            if (found==cells.end()) { reasons[i]=2; continue; }
            const Cell& cell=found->second;
            int s=cell.rows.size(); sizes[i]=s;
            if (s<2) { reasons[i]=3; continue; }
            double value=std::log(static_cast<double>(s)/n);
            reasons[i]=0;
            for (int j=0; j<d; ++j) {
                const Margin& margin=cell.margins[j];
                if (row[j]<margin.xi.front() || row[j]>margin.xi.back()) {
                    reasons[i]=4; break;
                }
                int a=std::max(0,static_cast<int>(std::lower_bound(margin.xi.begin(),margin.xi.end(),row[j])-margin.xi.begin())-1);
                if (!std::isfinite(margin.log_density[a])) { reasons[i]=5; break; }
                value+=margin.log_density[a];
            }
            if (reasons[i]==0) logf[i]=value;
        }
        return 0;
    } catch (...) { return 2; }
}

extern "C" void pss_v2_evaluate_r(double* x, int* n, int* d, int* ell,
                                  double* q, int* nq, double* logf, int* sizes,
                                  int* reasons, double* summary, int* status) {
    *status=pss_v2_evaluate(x,*n,*d,*ell,q,*nq,logf,sizes,reasons,summary);
}
