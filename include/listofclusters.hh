#ifndef _METRIC_LISTOFCLUSTERS_HH_
#define _METRIC_LISTOFCLUSTERS_HH_
#include <glob.hh>
#include <cluster.hh>
#include <resultslist.hh>
#include <internal_object.hh>

namespace metric
{

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
class listofclusters
{
public:
    typedef internal_object<object_t> internal_object_t;
    typedef resultslist<object_t> resultslist_t;
    typedef cluster<object_t> cluster_t;
    typedef std::vector<internal_object_t> supercluster_t;
    typedef std::list<std::vector<cluster_t>> list_t;

private:
    list_t _list;
    supercluster_t _leftovers;
    uint32_t _cid;
    std::map<std::pair<uint32_t,uint32_t>,std::pair<double,std::chrono::milliseconds>> _dcache;

public:
    listofclusters(void);
    listofclusters(const listofclusters&);
    listofclusters& operator=(const listofclusters&);
    ~listofclusters(void);

    void insert(const object_t&,const uint32_t&);
    void remove(const object_t&,const uint32_t&);

    resultslist_t knn_search(const object_t&,const uint32_t&,const size_t&);
    resultslist_t range_search(const object_t&,const uint32_t&,const double&);

private:
    void range_search(resultslist_t&,const double&);
    void knn_search(resultslist_t&,const size_t&);
    void explore(resultslist_t&,const cluster_t&,const double&);
    double internal_distance(const internal_object_t&,const internal_object_t&);

};
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>::listofclusters(void)
{
    this->_cid=0U;
    this->_leftovers.reserve(overflow);
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>::listofclusters(const listofclusters &_listofclusters)
{
    this->_cid=_listofclusters._cid;
    this->_list=_listofclusters._list;
    this->_leftovers=_listofclusters._leftovers;
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>& listofclusters<object_t,distance,bucket_size,overflow>::operator=(const listofclusters &_listofclusters)
{
    this->_cid=_listofclusters._cid;
    this->_list=_listofclusters._list;
    this->_leftovers=_listofclusters._leftovers;
    return(*this);
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>::~listofclusters(void)
{
    this->_list.clear();
    this->_leftovers.clear();
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::remove(const object_t &_object,const uint32_t &_id)
{
    double dist=0.0;
    bool exit=false;

    for(auto& clusters : this->_list)
        {
            for(auto& cluster : clusters)
                {
                    dist=distance(_object,cluster.centroid().object());
                    if(dist<=cluster.radius())
                        {
                            cluster.remove(_id);
                            if(cluster.empty())
                                {
                                    uint32_t id=cluster.id();
                                    clusters.erase(std::find_if(clusters.begin(),clusters.end(),[&id](const cluster_t &_cluster)->bool{return(_cluster.id()==id);}));
                                }
                            exit=true;
                            break;
                        }
                }
            if(exit) break;
        }

    if(!exit)
        this->_leftovers.erase(std::find_if(this->_leftovers.begin(),this->_leftovers.end(),[&_id](const internal_object_t &_object)->bool{return(_object.id()==_id);}));
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::insert(const object_t &_object,const uint32_t &_id)
{
    double dist=0.0;

    for(auto& clusters : this->_list)
        {
            for(auto& cluster : clusters)
                {
                    dist=distance(_object,cluster.centroid().object());
                    if(dist<=cluster.radius())
                        {
                            cluster.insert(_object,_id,dist);
                            return;
                        }
                }
        }

    this->_leftovers.push_back(internal_object_t(_object,_id));

    if(this->_leftovers.size()==overflow)
        {
            std::cout << "begin overflow" << std::endl;
            unsigned seed=std::chrono::system_clock::now().time_since_epoch().count();
            std::mt19937 rng(seed);
            std::uniform_int_distribution<int> uniform(0,overflow-1);

            typename supercluster_t::iterator it=this->_leftovers.begin();
            std::advance(it,uniform(rng));

            std::map<uint32_t,double> accum;
            std::vector<cluster<object_t>> clusters;
            uint32_t id=0U;

            do
                {
                    internal_object_t centroid=*it;
                    this->_leftovers.erase(it);

                    std::set<internal_object_t,compare<internal_object_t>> dists;
                    for(auto& object : this->_leftovers)
                        {
                            dist=distance(object.object(),centroid.object());
                            accum[object.id()]+=dist;
                            dists.insert(internal_object_t(object.object(),object.id(),dist));
                        }

                    cluster_t cluster(this->_cid++,centroid);
                    for(size_t i=0; i<bucket_size; i++)
                        {
                            id=dists.begin()->id();
                            cluster.insert(dists.begin()->object(),dists.begin()->id(),dists.begin()->distance());
                            cluster.radius(dists.begin()->distance());
                            accum.erase(accum.find(id));
                            this->_leftovers.erase(std::find_if(this->_leftovers.begin(),this->_leftovers.end(),[&id](const internal_object_t &_object)->bool{return(_object.id()==id);}));
                            dists.erase(dists.begin());
                        }

                    auto max=std::max_element(accum.begin(),accum.end(),[](const decltype(accum)::value_type &a,const decltype(accum)::value_type &b)->bool{return(a.second<b.second);});

                    id=max->first;
                    it=std::find_if(this->_leftovers.begin(),this->_leftovers.end(),[&id](const internal_object_t &_object)->bool{return(_object.id()==id);});
                    accum.erase(max);

                    clusters.push_back(cluster);

                }
            while(this->_leftovers.size()>bucket_size);
            this->_list.push_back(clusters);

            std::cout << "end overflow" << std::endl;
        }
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
typename listofclusters<object_t,distance,bucket_size,overflow>::resultslist_t listofclusters<object_t,distance,bucket_size,overflow>::range_search(const object_t &_object,const uint32_t &_id,const double &_radius)
{
    resultslist_t results(internal_object_t(_object,_id));
    double dist=0.0;
    this->range_search(results,_radius);

    for(auto& object : this->_leftovers)
        {
            dist=this->internal_distance(results.centroid(),object);
            if(dist<=_radius)
                results.push(object.object(),object.id(),dist);
        }

    this->_dcache.clear();
    return(results);
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::range_search(resultslist_t &_results,const double &_radius)
{
    double dist=0.0;
    bool exit=false;

    for(auto& clusters : this->_list)
        {
            for(auto& cluster : clusters)
                {
                    dist=this->internal_distance(_results.centroid(),cluster.centroid());

                    if((dist-_radius)<=cluster.radius())
                        {
                            if(dist<=_radius)
                                _results.push(cluster.centroid().object(),cluster.centroid().id(),dist);
                            this->explore(_results,cluster,_radius);
                        }

                    if((dist-(2.0*_radius))<=cluster.radius())
                        {
                            exit=true;
                            break;
                        }
                }
            if(exit)
                break;
        }
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::explore(resultslist_t &_results,const cluster_t &_cluster,const double &_radius)
{
    double dist=0.0;

    for(auto& object : _cluster.bucket())
        {
            dist=this->internal_distance(_results.centroid(),_cluster.centroid());
            if((dist-_radius)<=object.distance())
                {
                    dist=this->internal_distance(_results.centroid(),object);
                    if(dist<=_radius)
                        _results.push(object.object(),object.id(),dist);
                }
        }
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
typename listofclusters<object_t,distance,bucket_size,overflow>::resultslist_t listofclusters<object_t,distance,bucket_size,overflow>::knn_search(const object_t &_object,const uint32_t &_id,const size_t &_k)
{
    resultslist_t results(internal_object_t(_object,_id),_k);
    double radius=std::numeric_limits<double>::max();
    double dist=0.0,diff=0.0;
    bool exit=false;

    for(auto& clusters : this->_list)
        {
            for(auto& cluster : clusters)
                {
                    dist=this->internal_distance(cluster.centroid(),results.centroid());
                    if(dist<cluster.radius())
                        {
                            radius=cluster.radius()-dist;
                            exit=true;
                            break;
                        }
                    else
                        {
                            diff=dist-cluster.radius();
                            if(diff<radius)
                                radius=diff;
                        }
                }
            if(exit)
                break;
        }

    do
        {
            this->range_search(results,radius);
            if(radius==std::numeric_limits<double>::max() || radius==0.0) break;
            radius+=radius*ALFA;
        }
    while(results.results().size()<_k);

    for(auto& object : this->_leftovers)
        {
            dist=this->internal_distance(results.centroid(),object);
            if(dist<=radius)
                results.push(object.object(),object.id(),dist);
        }

    this->_dcache.clear();
    return(results);
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
double listofclusters<object_t,distance,bucket_size,overflow>::internal_distance(const internal_object_t &_a,const internal_object_t &_b)
{
    std::pair<uint32_t,uint32_t> c(_a.id(),_b.id());

    if(this->_dcache.size()>MAX_DCACHE_SIZE)
        this->_dcache.erase(std::min_element(this->_dcache.begin(),this->_dcache.end(),[](typename decltype(this->_dcache)::value_type &a,typename decltype(this->_dcache)::value_type& b)->bool{return(a.second<b.second);}));

    if(!this->_dcache.count(c))
        {
            double dist=distance(_a.object(),_b.object());
            auto timestamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
            this->_dcache[c]=std::pair<double,std::chrono::milliseconds>(dist,timestamp);
        }

    return(this->_dcache[c].first);
}

};
#endif
