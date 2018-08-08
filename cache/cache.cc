#include <listofclusters.hh>
#include <armadillo>
#include <iostream>
#include <random>
#include <chrono>
#include <math.h>
#include <resultslist.hh>

double idistance(arma::rowvec a,arma::rowvec b){
	arma::rowvec c=a-b;
	return(sqrt(arma::dot(c,c)));
}
double cdistance(metric::resultslist<arma::rowvec> a,metric::resultslist<arma::rowvec> b){
	arma::rowvec c=a.centroid().object()-b.centroid().object();
	return(sqrt(arma::dot(c,c)));
}

int main(int argc,char** argv){
	arma::mat data;
	data.load(argv[1],arma::csv_ascii);
	metric::listofclusters<arma::rowvec,idistance,50,10000> loc;

	uint32_t id=0U;

	auto start=std::chrono::system_clock::now();
	std::ifstream fdb(argv[2]);
	while(fdb >> id)
		loc.insert(data.row(id));
	fdb.close();
	auto end=std::chrono::system_clock::now();
	auto elapsed=std::chrono::duration_cast<std::chrono::seconds>(end-start);
	std::cout << "indexing time : " << elapsed.count() << " seconds " << std::endl;

	std::ifstream fqueries(argv[3]);
	std::vector<uint32_t> queries;
	while(fqueries >> id)
		queries.push_back(id);
	fqueries.close();
	
	for(size_t i=0;i<queries.size();++i){
		auto start=std::chrono::system_clock::now();
		auto results=loc.knn_search(data.row(queries[i]),queries[i],100);
		auto end=std::chrono::system_clock::now();
		auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
		std::cout << "searching time query " << queries[i] << " : " << elapsed.count() << " milliseconds " << std::endl;
	}

	return(0);
}
