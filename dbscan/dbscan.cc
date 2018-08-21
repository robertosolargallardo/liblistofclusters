#include <listofclusters.hh>
#include <resultslist.hh>
#include <armadillo>
#include <iostream>
#include <random>
#include <chrono>
#include <memory>
#include <math.h>

typedef arma::rowvec sift_t;

double idistance(sift_t a,sift_t b)
{
    sift_t c=a-b;
    return(sqrt(arma::dot(c,c)));
}
arma::mat DB;

class dbscan{
	private:	metric::listofclusters<sift_t,idistance,10,1000> _index;
				std::vector<uint32_t> _ids;
				std::map<uint32_t,std::vector<uint32_t>> _clusters;
				std::map<uint32_t,bool>                  _mark;

	public: 	dbscan(void){
				
				}
			 	dbscan(const dbscan &_dbscan){
					this->_index=_dbscan._index;
					this->_ids=_dbscan._ids;
				}
			 	dbscan& operator=(const dbscan &_dbscan){
					this->_index=_dbscan._index;
					this->_ids=_dbscan._ids;
					return(*this);
				}
				~dbscan(void){
					;
				}
				dbscan(const std::string &_filename){
    				uint32_t id=0U;
					std::ifstream ftrain(_filename);
    				while(ftrain >> id){
						this->_ids.push_back(id);
						this->_index.insert(DB.row(id),id);
					}
    				ftrain.close();
				}

				void scan(const double &_epsilon,const size_t &_min_size){
					unsigned seed=std::chrono::system_clock::now().time_since_epoch().count();
					std::mt19937 rng(seed);
					uint32_t id=0;
					size_t fail=0;

					while(this->_ids.size()!=fail){
						std::uniform_int_distribution<size_t> uniform(0,this->_ids.size()-1);
						id=this->_ids[uniform(rng)];
						auto query=this->_index.range_search(DB.row(id),id,_epsilon);

						if(query.results().size()>=_min_size){
							fail=0;
							std::cout << id << std::endl;
							this->_index.remove(DB.row(id),id);
						   this->_ids.erase(std::find_if(this->_ids.begin(),this->_ids.end(),[id](const uint32_t _id)->bool{return(_id==id);}));

							for(auto result : query.results()){
								std::cout << result.id() << std::endl;
								this->_index.remove(DB.row(result.id()),result.id());
						      this->_ids.erase(std::find_if(this->_ids.begin(),this->_ids.end(),[id=result.id()](const uint32_t _id)->bool{return(_id==id);}));
							}
							std::cout << "SIZE : " << this->_ids.size() << std::endl;
							/*this->_mark[id]=true;
					      this->_ids.erase(std::find_if(this->_ids.begin(),this->_ids.end(),[id](const uint32_t _id)->bool{return(_id==id);}));

							for(auto result : query.results()){
								if(!this->_mark.count(result.id())){
									this->_clusters[id].push_back(result.id());
									this->_mark[result.id()]=true;
						      	this->_ids.erase(std::find_if(this->_ids.begin(),this->_ids.end(),[id=result.id()](const uint32_t _id)->bool{return(_id==id);}));
								}
							}

							this->scan(id,_epsilon,_min_size);

							std::cout << this->_clusters[id].size() << std::endl;
							std::cout << "SIZE : " << this->_ids.size() << std::endl;
							getchar();*/
						}
						else fail++;
					}
				}
				void scan(const uint32_t &_centroid,const double &_epsilon,const size_t &_min_size){
					size_t N=this->_clusters[_centroid].size();
					uint32_t id=0U;

					for(size_t i=0;i<N;++i){
						id=this->_clusters[_centroid][i];
						auto query=this->_index.range_search(DB.row(id),id,_epsilon);
						
						if(query.results().size()>=_min_size){
							for(auto result : query.results()){
								if(!this->_mark.count(result.id())){
									this->_clusters[_centroid].push_back(result.id());
									this->_mark[result.id()]=true;
						         this->_ids.erase(std::find_if(this->_ids.begin(),this->_ids.end(),[id=result.id()](const uint32_t _id)->bool{return(_id==id);}));
									++N;
								}
							}	
						}
					}
				}
};
int main(int argc,char** argv)
{
    DB.load(argv[1],arma::csv_ascii);
	 dbscan db(argv[2]);

	 db.scan(250.0,10);
    
    return(0);
}
