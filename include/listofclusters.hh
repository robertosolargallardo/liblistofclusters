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
    typedef std::list<std::vector<cluster_t>> list_t;

private:
    list_t     _list;
    cluster_t  _supercluster;
    uint32_t   _cid;
    std::map<uint32_t,std::map<uint32_t,double>> _dcache;

public:
    listofclusters(void);
    listofclusters(const listofclusters&);
    listofclusters& operator=(const listofclusters&);
    ~listofclusters(void);

    void insert(const object_t&,const uint32_t&);
    void remove(const object_t&,const uint32_t&);

    resultslist_t knn_search(const object_t&,const uint32_t&,const size_t&);
    resultslist_t range_search(const object_t&,const uint32_t&,const double&);

    void show(void)
    {
        /*************/

        for(auto& clusters : this->_list)
            {
                for(auto& cluster : clusters)
                    std::cout << cluster.id() << " " << cluster.size() << " " << cluster.radius() << std::endl;
            }
        /*************/

    }

private:
    void range_search(resultslist_t&,const double&);
    void knn_search(resultslist_t&,const size_t&);
    void explore(resultslist_t&,const cluster_t&,const double&);
    double internal_distance(const internal_object_t&,const internal_object_t&);

};
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>::listofclusters(void)
{
    this->_cid=1U;
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>::listofclusters(const listofclusters &_listofclusters)
{
    this->_cid=_listofclusters._cid;
    this->_list=_listofclusters._list;
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>& listofclusters<object_t,distance,bucket_size,overflow>::operator=(const listofclusters &_listofclusters)
{
    this->_cid=_listofclusters._cid;
    this->_list=_listofclusters._list;
    return(*this);
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
listofclusters<object_t,distance,bucket_size,overflow>::~listofclusters(void)
{
    this->_list.clear();
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
        this->_supercluster.remove(_id);
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

    if(this->_supercluster.empty())
        this->_supercluster=cluster_t(SUPERCLUSTER,internal_object_t(_object,_id));
    else
        {
            dist=distance(_object,this->_supercluster.centroid().object());
            this->_supercluster.insert(_object,_id,dist);
        }

    if(this->_supercluster.size()==overflow)
        {
            std::map<uint32_t,double> accum;
            std::vector<cluster_t> clusters;
            clusters.reserve(size_t(std::ceil(double(overflow)/double(bucket_size))));

            do
                {
                    cluster_t cluster(this->_cid++,this->_supercluster.centroid());
                    typename cluster_t::bucket_t bucket=this->_supercluster.bucket();
                    std::for_each(bucket.begin(),bucket.end(),[&accum](const internal_object_t &_object)->void{accum[_object.id()]+=_object.distance();});
                    auto it=bucket.begin();
                    std::advance(it,bucket_size);
                    cluster.radius(it->distance());
                    clusters.push_back(cluster);

                    for(size_t i=0; i<bucket_size && !bucket.empty(); ++i)
                        {
                            auto object=*bucket.begin();
                            for(size_t j=0; j<clusters.size(); ++j)
                                {
                                    dist=this->internal_distance(object,clusters[j].centroid());
                                    if(dist<=clusters[j].radius())
                                        {
                                            clusters[j].insert(object.object(),object.id(),object.distance());
                                            break;
                                        }
                                }
                            accum.erase(accum.find(object.id()));
                            bucket.erase(bucket.begin());
                        }
                    this->_supercluster.clear();

                    if(bucket.empty()) break;

                    auto max=std::max_element(accum.begin(),accum.end(),[](const decltype(accum)::value_type &a,const decltype(accum)::value_type &b)->bool{return(a.second<b.second);});
                    auto centroid=std::find_if(bucket.begin(),bucket.end(),[id=max->first](const internal_object_t &_object)->bool{return(_object.id()==id);});

                    this->_supercluster=cluster_t(SUPERCLUSTER,*centroid);
                    accum.erase(max);
                    bucket.erase(centroid);

                    for(auto& object : bucket)
                        {
                            dist=this->internal_distance(object,this->_supercluster.centroid());
                            this->_supercluster.insert(object.object(),object.id(),dist);
                        }
                }
            while(this->_supercluster.size()>bucket_size);

            this->_list.push_back(clusters);
            this->_dcache.clear();

        }
}

template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
typename listofclusters<object_t,distance,bucket_size,overflow>::resultslist_t listofclusters<object_t,distance,bucket_size,overflow>::range_search(const object_t &_object,const uint32_t &_id,const double &_radius)
{
    resultslist_t results(internal_object_t(_object,_id));
    this->range_search(results,_radius);
    this->_dcache.erase(this->_dcache.find(_id));
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

    if(!exit)
        {
            dist=this->internal_distance(_results.centroid(),this->_supercluster.centroid());
            if((dist-_radius)<=this->_supercluster.radius())
                {
                    if(dist<=_radius)
                        _results.push(this->_supercluster.centroid().object(),this->_supercluster.centroid().id(),dist);
                    this->explore(_results,this->_supercluster,_radius);
                }
        }


}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
void listofclusters<object_t,distance,bucket_size,overflow>::explore(resultslist_t &_results,const cluster_t &_cluster,const double &_radius)
{
    double dist=0.0;
    double dqc=this->internal_distance(_results.centroid(),_cluster.centroid());

    for(auto& object : _cluster.bucket())
        {
            if((dqc-_radius)<=object.distance())
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
                    diff=cluster.radius()-dist;
                    if(diff>0.0)
                        {
                            radius=diff;
                            exit=true;
                            break;
                        }
                    else
                        {
                            if((-diff)<radius)
                                radius=(-diff);
                        }
                }
            if(exit)
                break;
        }

    do
        {
            this->range_search(results,radius);

            /***************************/
            if(_id==552511)
                std::cout << radius << std::endl;
            /***************************/

            if(radius==std::numeric_limits<double>::max() || radius==0.0) break;
            radius+=radius*ALFA;
        }
    while(results.results().size()<_k);

    this->_dcache.erase(this->_dcache.find(_id));
    return(results);
}
template <class object_t,double (*distance)(object_t,object_t),size_t bucket_size,size_t overflow>
double listofclusters<object_t,distance,bucket_size,overflow>::internal_distance(const internal_object_t &_a,const internal_object_t &_b)
{
    if(!this->_dcache.count(_a.id()))
        this->_dcache[_a.id()]=std::map<uint32_t,double>();
    if(!this->_dcache[_a.id()].count(_b.id()))
        this->_dcache[_a.id()][_b.id()]=distance(_a.object(),_b.object());

    return(this->_dcache[_a.id()][_b.id()]);
}

};
#endif
