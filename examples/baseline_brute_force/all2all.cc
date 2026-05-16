// Brute-force O(n^2) pairwise distance baseline. Does not use the LC index;
// kept for benchmarking and as a sanity-check reference.
#include <armadillo>
#include <iostream>
#include <cmath>

typedef arma::rowvec sift_t;
arma::mat DB;

[[nodiscard]] static double idistance(const sift_t &a, const sift_t &b) noexcept
{
    const sift_t c = a - b;
    return std::sqrt(arma::dot(c, c));
}
int main(int argc,char** argv)
{
    DB.load(argv[1],arma::csv_ascii);

	 for(size_t i=0;i<DB.n_rows;i++){
	 	for(size_t j=i+1;j<DB.n_rows;j++){
			std::cout << i << " " << j << " " << idistance(DB.row(i),DB.row(j)) << std::endl;
		}		
    }
    return(0);
}
