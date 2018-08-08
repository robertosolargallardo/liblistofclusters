#include <listofclusters.hh>
#include <resultslist.hh>
#include <armadillo>
#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <math.h>

#define IS_K 100
#define CS_K 10
#define CS_N 1000

typedef arma::rowvec sift_t;

double idistance(sift_t a,sift_t b){
	sift_t c=a-b;
	return(sqrt(arma::dot(c,c)));
}
double cdistance(metric::resultslist<sift_t> a,metric::resultslist<sift_t> b){
	sift_t c=a.centroid().object()-b.centroid().object();
	return(sqrt(arma::dot(c,c)));
}

arma::mat DB;

class index_service{
	private:	metric::listofclusters<sift_t,idistance,50,10000> _index;
	
	public:	index_service(void){
					;
				}
				index_service(const std::string &_filename){
					uint32_t id=0U;
					auto start=std::chrono::system_clock::now();
					std::ifstream fdb(_filename);
					while(fdb >> id)
						this->_index.insert(DB.row(id),id);
					fdb.close();
					auto end=std::chrono::system_clock::now();
					auto elapsed=std::chrono::duration_cast<std::chrono::seconds>(end-start);
					std::cout << "indexing time : " << elapsed.count() << " seconds " << std::endl;

				}
				index_service(const index_service &_is){
					this->_index=_is._index;
				}
				index_service& operator=(const index_service &_is){
					this->_index=_is._index;
					return(*this);
				}
				~index_service(void){
					;
				}
			
				metric::resultslist<sift_t> search(const sift_t &_query,const uint32_t &_idq){
					return(this->_index.knn_search(_query,_idq,IS_K));
				}
};
class cache_service{
	private: metric::listofclusters<metric::resultslist<sift_t>,cdistance,10,100> _index;
				std::map<uint32_t,std::chrono::milliseconds>   _cache;
				std::map<uint32_t,metric::resultslist<sift_t>> _objects;

	public:  cache_service(void){
					;
				}
				cache_service(const cache_service &_cs){
					this->_index=_cs._index;
					this->_cache=_cs._cache;
				}
				cache_service& operator=(const cache_service &_cs){
					this->_index=_cs._index;
					this->_cache=_cs._cache;
					return(*this);
				}
				~cache_service(void){
					this->_cache.clear();
				}
				metric::resultslist<sift_t> search(const sift_t &_query,const uint32_t &_idq){
					std::cout << "cache search" << std::endl;
				
					metric::resultslist<sift_t> result;
					metric::resultslist<sift_t> query(metric::internal_object<sift_t>(_query,_idq),CS_K);

					auto results=this->_index.knn_search(query,_idq,CS_K);
					std::cout << "k : " << results.results().size() << std::endl;
					return(result);	
				}
				bool evaluate(const metric::resultslist<sift_t> &_cached){
					return(false);
				}
				void insert(const metric::resultslist<sift_t> &_result){
					std::cout << "cache insert :" << this->_cache.size() <<std::endl;
					this->_index.insert(_result,_result.centroid().id());
					this->_cache[_result.centroid().id()]=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
					this->_objects[_result.centroid().id()]=_result;

					if(this->_cache.size()>CS_N){
						auto min=std::min_element(this->_cache.begin(),this->_cache.end(),[](decltype(this->_cache)::value_type &l,decltype(this->_cache)::value_type& r)->bool{return(l.second<r.second);});
						uint32_t id=min->first;
						this->_index.remove(this->_objects[id],id);
						this->_objects.erase(std::find_if(this->_objects.begin(),this->_objects.end(),[&id](decltype(this->_objects)::value_type &a)->bool{return(a.first==id);}));
						this->_cache.erase(min);
					}
				}
};

int main(int argc,char** argv){
	DB.load(argv[1],arma::csv_ascii);

	std::shared_ptr<index_service> is=std::make_shared<index_service>(argv[2]);
	std::shared_ptr<cache_service> cs=std::make_shared<cache_service>();

	std::ifstream fqueries(argv[3]);
	std::vector<uint32_t> queries;
	uint32_t id=0U;
	while(fqueries >> id)
		queries.push_back(id);
	fqueries.close();

	for(size_t i=0;i<queries.size();++i){
		metric::resultslist<sift_t> cached=cs->search(DB.row(queries[i]),queries[i]);
		bool hit=cs->evaluate(cached);

		if(!hit){
			metric::resultslist<sift_t> result=is->search(DB.row(queries[i]),queries[i]);
			cs->insert(result);
		}
	}
	return(0);
}
