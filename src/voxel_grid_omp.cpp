//
// Created by hk on 3/26/23.
//
#include "voxel_grid_omp.h"
//#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
//#include <pcl/common/common.h>
//#include <Eigen/Dense>

//#include <unordered_map>
#include <unordered_set>

#include <omp.h>
//template <typename PointT>
void
pcl::VoxelGridOMP::setNumberOfThreads(unsigned int nr_threads) {
    if (nr_threads == 0)
#ifdef _OPENMP
        threads_ = omp_get_num_procs();
#else
        threads_ = 1;
#endif
    else
        threads_ = nr_threads;
    // printf("Voxel Grid OMP set number of threads: %d.\n", threads_);
}

//struct Voxel
//{
//    pcl::PointXYZINormal centroid;
//    unsigned int num_points;
//
//    Voxel() : num_points(0), centroid(pcl::PointXYZINormal()) {}
//    Voxel(const unsigned int n, const pcl::PointXYZINormal& p) : num_points(n), centroid(p) {}
//};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::VoxelGridOMP::applyFilter(PointCloud& output) {
    //    TicToc t_bbox;
    // Has the input dataset been set already?
    if (!input_) {
        PCL_WARN("[pcl::%s::applyFilter] No input dataset given!\n", getClassName().c_str());
        output.width = output.height = 0;
        output.points.clear();
        return;
    }

    // Copy the header (and thus the frame_id) + allocate enough space for points
    output.height = 1;                    // downsampling breaks the organized structure
    output.is_dense = true;                 // we filter out invalid points

    Eigen::Vector4f min_p, max_p;
    // Get the minimum and maximum dimensions
    if (!filter_field_name_.empty()) // If we don't want to process the entire cloud...
        getMinMax3D<PointT>(input_, *indices_, filter_field_name_, static_cast<float> (filter_limit_min_), static_cast<float> (filter_limit_max_), min_p, max_p, filter_limit_negative_);
    else {
        //        getMinMax3D<PointT>(*input_, *indices_, min_p, max_p);
        getMinMax3DOMP(*input_, *indices_, min_p, max_p);
        //        printf("min max OMP cost: %fms\n", t_bbox.toc());
    }

    // Check that the leaf size is not too small, given the size of the data
    int64_t dx = static_cast<int64_t>((max_p[0] - min_p[0]) * inverse_leaf_size_[0]) + 1;
    int64_t dy = static_cast<int64_t>((max_p[1] - min_p[1]) * inverse_leaf_size_[1]) + 1;
    int64_t dz = static_cast<int64_t>((max_p[2] - min_p[2]) * inverse_leaf_size_[2]) + 1;

    if ((dx * dy * dz) > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        PCL_WARN("[pcl::%s::applyFilter] Leaf size is too small for the input dataset. Integer indices would overflow.", getClassName().c_str());
        output = *input_;
        return;
    }

    // Compute the minimum and maximum bounding box values
    min_b_[0] = static_cast<int> (floor(min_p[0] * inverse_leaf_size_[0]));
    max_b_[0] = static_cast<int> (floor(max_p[0] * inverse_leaf_size_[0]));
    min_b_[1] = static_cast<int> (floor(min_p[1] * inverse_leaf_size_[1]));
    max_b_[1] = static_cast<int> (floor(max_p[1] * inverse_leaf_size_[1]));
    min_b_[2] = static_cast<int> (floor(min_p[2] * inverse_leaf_size_[2]));
    max_b_[2] = static_cast<int> (floor(max_p[2] * inverse_leaf_size_[2]));

    // Compute the number of divisions needed along all axis
    div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
    div_b_[3] = 0;

    // Set up the division multiplier
    divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);
    //    printf("bbox cost: %fms\n", t_bbox.toc());


    int centroid_size = 4;
    if (downsample_all_data_) {
        centroid_size = boost::mpl::size<FieldList>::value;
        //        printf("downsample_all_data_\n");
    }

    // ---[ RGB special case
    std::vector<pcl::PCLPointField> fields;
    int rgba_index = -1;
    rgba_index = pcl::getFieldIndex(*input_, "rgb", fields);
    if (rgba_index == -1)
        rgba_index = pcl::getFieldIndex(*input_, "rgba", fields);
    if (rgba_index >= 0) {
        rgba_index = fields[rgba_index].offset;
        centroid_size += 3;
    }

    TicToc t_p2v;
    //global variables
    std::vector<PointCloudPtr> cloud_all_threads(threads_);
    for (size_t i = 0; i < threads_; ++i) {
        cloud_all_threads[i].reset(new PointCloud());
    }

#pragma omp parallel num_threads(threads_)
    {
        int thread_id = omp_get_thread_num();
        PointCloudPtr& cloud_thread = cloud_all_threads[thread_id];
        //        PointCloud cloud_thread;
        std::vector<cloud_point_index_idx> index_vector;
        index_vector.reserve(indices_->size() / threads_);

        //        int num_points = 0;
                // If we don't want to process the entire cloud, but rather filter points far away from the viewpoint first...
        if (!filter_field_name_.empty()) {
            // Get the distance field index
            std::vector<pcl::PCLPointField> fields;
            int distance_idx = pcl::getFieldIndex(*input_, filter_field_name_, fields);
            if (distance_idx == -1)
                PCL_WARN("[pcl::%s::applyFilter] Invalid filter field name. Index is %d.\n", getClassName().c_str(),
                    distance_idx);

            // First pass: go over all points and insert them into the index_vector vector
            // with calculated idx. Points with the same idx value will contribute to the
            // same point of resulting CloudPoint

            int thread_id = omp_get_thread_num();
            //#pragma omp parallel
            for (std::vector<int>::const_iterator it = indices_->begin(); it != indices_->end(); it += thread_id) {
                if (!input_->is_dense)
                    // Check if the point is invalid
                    if (!pcl_isfinite(input_->points[*it].x) ||
                        !pcl_isfinite(input_->points[*it].y) ||
                        !pcl_isfinite(input_->points[*it].z))
                        continue;

                // Get the distance value
                const uint8_t* pt_data = reinterpret_cast<const uint8_t*> (&input_->points[*it]);
                float distance_value = 0;
                memcpy(&distance_value, pt_data + fields[distance_idx].offset, sizeof(float));

                if (filter_limit_negative_) {
                    // Use a threshold for cutting out points which inside the interval
                    if ((distance_value < filter_limit_max_) && (distance_value > filter_limit_min_))
                        continue;
                }
                else {
                    // Use a threshold for cutting out points which are too close/far away
                    if ((distance_value > filter_limit_max_) || (distance_value < filter_limit_min_))
                        continue;
                }

                int ijk0 = static_cast<int> (floor(input_->points[*it].x * inverse_leaf_size_[0]) -
                    static_cast<float> (min_b_[0]));
                int ijk1 = static_cast<int> (floor(input_->points[*it].y * inverse_leaf_size_[1]) -
                    static_cast<float> (min_b_[1]));
                int ijk2 = static_cast<int> (floor(input_->points[*it].z * inverse_leaf_size_[2]) -
                    static_cast<float> (min_b_[2]));

                // Compute the centroid leaf index
                int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
                index_vector.push_back(cloud_point_index_idx(static_cast<unsigned int> (idx), *it));
            }
        }
        // No distance filtering, process all data
        else {
            // First pass: go over all points and insert them into the index_vector vector
            // with calculated idx. Points with the same idx value will contribute to the
            // same point of resulting CloudPoint

//        printf("check 1\n");
#pragma omp for
//            for (std::vector<int>::const_iterator it = indices_->begin(); it != indices_->end(); ++it) {
            for (size_t i = 0; i < indices_->size(); ++i) {
                const int& point_id = indices_->at(i);
                const PointT& point = input_->points[point_id];
                if (!input_->is_dense)
                    // Check if the point is invalid
                    if (!pcl_isfinite(point.x) ||
                        !pcl_isfinite(point.y) ||
                        !pcl_isfinite(point.z))
                        continue;

                int ijk0 = static_cast<int> (floor(point.x * inverse_leaf_size_[0]) -
                    static_cast<float> (min_b_[0]));
                int ijk1 = static_cast<int> (floor(point.y * inverse_leaf_size_[1]) -
                    static_cast<float> (min_b_[1]));
                int ijk2 = static_cast<int> (floor(point.z * inverse_leaf_size_[2]) -
                    static_cast<float> (min_b_[2]));

                // Compute the centroid leaf index
                int idx = ijk0 * divb_mul_[0] + ijk1 * divb_mul_[1] + ijk2 * divb_mul_[2];
                /// !!! so slow  !!!
                index_vector.emplace_back(
                    cloud_point_index_idx(static_cast<unsigned int> (idx), point_id));
            }
        }
        //        printf("thread %d points to voxel cost: %fms\n", thread_id, t_p2v.toc());

                /// !!! too slow!!!
        //        TicToc t_sort;
                // Second pass: sort the index_vector vector using value representing target cell as index
                // in effect all points belonging to the same output cell will be next to each other
        std::sort(index_vector.begin(), index_vector.end(), std::less<cloud_point_index_idx>());
        //        printf("thread %d sort cost: %fms\n", thread_id, t_sort.toc());

        //    TicToc t_valid_voxel;
                // Third pass: count output cells
                // we need to skip all the same, adjacenent idx values
        unsigned int total = 0;
        unsigned int index = 0;
        // first_and_last_indices_vector[i] represents the index in index_vector of the first point in
        // index_vector belonging to the voxel which corresponds to the i-th output point,
        // and of the first point not belonging to.
        std::vector<std::pair<unsigned int, unsigned int> > first_and_last_indices_vector;
        // Worst case size
        first_and_last_indices_vector.reserve(index_vector.size());

        while (index < index_vector.size()) {
            unsigned int i = index + 1;
            while (i < index_vector.size() && index_vector[i].idx == index_vector[index].idx)
                ++i;
            //todo
            if (i - index >= min_points_per_voxel_) {
                ++total;
                first_and_last_indices_vector.push_back(std::pair<unsigned int, unsigned int>(index, i));
            }
            index = i;
        }
        //        printf("thread %d compute valid voxel cost: %fms\n", thread_id, t_valid_voxel.toc());

                //    TicToc t_layout;
                // Fourth pass: compute centroids, insert them into their final position
        //        cloud_thread.points.resize(total);
        cloud_thread->points.resize(total);
        //        printf("thread %d resize cloud.\n", thread_id);
        //#pragma omp barrier

        //        if (save_leaf_layout_) {
        //            try {
        //                // Resizing won't reset old elements to -1.  If leaf_layout_ has been used previously, it needs to be re-initialized to -1
        //                uint32_t new_layout_size = div_b_[0] * div_b_[1] * div_b_[2];
        //                //This is the number of elements that need to be re-initialized to -1
        //                uint32_t reinit_size = std::min(static_cast<unsigned int> (new_layout_size),
        //                                                static_cast<unsigned int> (leaf_layout_.size()));
        //                for (uint32_t i = 0; i < reinit_size; i++) {
        //                    leaf_layout_[i] = -1;
        //                }
        //                leaf_layout_.resize(new_layout_size, -1);
        //            }
        //            catch (std::bad_alloc &) {
        //                throw PCLException("VoxelGrid bin size is too low; impossible to allocate memory for layout",
        //                                   "voxel_grid.hpp", "applyFilter");
        //            }
        //            catch (std::length_error &) {
        //                throw PCLException("VoxelGrid bin size is too low; impossible to allocate memory for layout",
        //                                   "voxel_grid.hpp", "applyFilter");
        //            }
        //        }
        //    printf("save layout cost: %fms\n", t_layout.toc());

        //        index = 0;
        //    }
        //    TicToc t_centriod;
        //#pragma omp parallel num_threads(threads_)
        //    {
        Eigen::VectorXf centroid = Eigen::VectorXf::Zero(centroid_size);
        Eigen::VectorXf temporary = Eigen::VectorXf::Zero(centroid_size);
        //        std::unordered_map<int, Voxel> voxels;
        //#pragma omp for
        for (unsigned int cp = 0; cp < first_and_last_indices_vector.size(); ++cp) {
            // calculate centroid - sum values from all input points, that have the same idx value in index_vector array
            unsigned int first_index = first_and_last_indices_vector[cp].first;
            unsigned int last_index = first_and_last_indices_vector[cp].second;
            //            int voxel_id = index_vector[first_index].idx;
            if (!downsample_all_data_) {
                centroid[0] = input_->points[index_vector[first_index].cloud_point_index].x;
                centroid[1] = input_->points[index_vector[first_index].cloud_point_index].y;
                centroid[2] = input_->points[index_vector[first_index].cloud_point_index].z;
            }
            else {
                // ---[ RGB special case
                if (rgba_index >= 0) {
                    // Fill r/g/b data, assuming that the order is BGRA
                    pcl::RGB rgb;
                    // copy memory from input to rgb
                    memcpy(&rgb,
                        reinterpret_cast<const char*> (&input_->points[index_vector[first_index].cloud_point_index]) +
                        rgba_index, sizeof(RGB));
                    centroid[centroid_size - 3] = rgb.r;
                    centroid[centroid_size - 2] = rgb.g;
                    centroid[centroid_size - 1] = rgb.b;
                }
                pcl::for_each_type<FieldList>(
                    NdCopyPointEigenFunctor<PointT>(input_->points[index_vector[first_index].cloud_point_index],
                        centroid));
            }

            for (unsigned int i = first_index + 1; i < last_index; ++i) {
                if (!downsample_all_data_) {
                    centroid[0] += input_->points[index_vector[i].cloud_point_index].x;
                    centroid[1] += input_->points[index_vector[i].cloud_point_index].y;
                    centroid[2] += input_->points[index_vector[i].cloud_point_index].z;
                }
                else {
                    // ---[ RGB special case
                    if (rgba_index >= 0) {
                        // Fill r/g/b data, assuming that the order is BGRA
                        pcl::RGB rgb;
                        memcpy(&rgb,
                            reinterpret_cast<const char*> (&input_->points[index_vector[i].cloud_point_index]) +
                            rgba_index, sizeof(RGB));
                        temporary[centroid_size - 3] = rgb.r;
                        temporary[centroid_size - 2] = rgb.g;
                        temporary[centroid_size - 1] = rgb.b;
                    }
                    // copying data between an Eigen type and a PointT (input point, output eigen)
                    pcl::for_each_type<FieldList>(
                        NdCopyPointEigenFunctor<PointT>(input_->points[index_vector[i].cloud_point_index],
                            temporary));
                    centroid += temporary;
                }
            }

            // index is centroid final position in resulting PointCloud
            if (save_leaf_layout_)
                leaf_layout_[index_vector[first_index].idx] = cp;
            //todo 记录点数
            centroid /= static_cast<float> (last_index - first_index);

            // store centroid
            // Do we need to process all the fields?
            if (!downsample_all_data_) {
                //            output.points[index].x = centroid[0];
                //            output.points[index].y = centroid[1];
                //            output.points[index].z = centroid[2];
                cloud_thread->points[cp].x = centroid[0];
                cloud_thread->points[cp].y = centroid[1];
                cloud_thread->points[cp].z = centroid[2];
            }
            else {
                //  NdCopyEigenPointFunctor( p1 the input Eigen type, p2 the output Point type)
                pcl::for_each_type<FieldList>(pcl::NdCopyEigenPointFunctor<PointT>(centroid, cloud_thread->points[cp]));
                // ---[ RGB special case
                if (rgba_index >= 0) {
                    // pack r/g/b into rgb
                    float r = centroid[centroid_size - 3], g = centroid[centroid_size - 2], b = centroid[centroid_size -
                        1];
                    int rgb = (static_cast<int> (r) << 16) | (static_cast<int> (g) << 8) | static_cast<int> (b);
                    //                memcpy (reinterpret_cast<char*> (&output.points[index]) + rgba_index, &rgb, sizeof (float));
                    memcpy(reinterpret_cast<char*> (&cloud_thread->points[cp]) + rgba_index, &rgb, sizeof(float));
                }
                //                Voxel vox_new(last_index - first_index, cloud_thread->points[cp]);
                //                voxels[voxel_id] = vox_new;
            }
            //        ++index;
        }
        //        printf("thread %d compute %d centroids cost: %fms\n", thread_id, (int)cloud_thread->points.size(), t_centriod.toc());
    }

    //merge cloud from all threads
    TicToc t_col;
    for (size_t i = 1; i < threads_; ++i) {
        *cloud_all_threads[0] += *cloud_all_threads[i];
    }
    //    printf("collect cloud %d from all thread cost: %fms\n", (int)cloud_all_threads[0]->size(), t_col.toc());

    if (final_filter) {
        // Create the filtering object
        pcl::VoxelGrid<PointT> vg;
        vg.setInputCloud(cloud_all_threads[0]);
        vg.setLeafSize(leaf_size_);

        TicToc t_vg;
        vg.filter(output);
        //        printf("final voxel grid PCL cost: %fms\n", t_vg.toc());
        //        std::cerr << "PointCloud after filtering: " << output.size() << std::endl;
    }
    else
        pcl::copyPointCloud(*cloud_all_threads[0], output);
}

void pcl::VoxelGridOMP::getMinMax3DOMP(const pcl::PointCloud<PointT>& cloud, const std::vector<int>& indices,
    Eigen::Vector4f& min_pt, Eigen::Vector4f& max_pt) {
    //    TicToc t_para;
        // prepare momery for all threads
    std::vector<Eigen::Vector4f> voxel_mins(threads_);
    std::vector<Eigen::Vector4f> voxel_maxs(threads_);

    // If the data is dense, we don't need to check for NaN
    if (cloud.is_dense) {
#pragma omp parallel num_threads(threads_)
        {
            // thread private
            int thread_id = omp_get_thread_num();
            Eigen::Vector4f voxel_min_t, voxel_max_t;
            voxel_min_t.setConstant(FLT_MAX);
            voxel_max_t.setConstant(-FLT_MAX);

#pragma omp for
            for (size_t i = 0; i < indices.size(); i++) {
                pcl::Array4fMapConst pt = cloud.points[indices[i]].getArray4fMap();
                voxel_min_t = voxel_min_t.array().min(pt);
                voxel_max_t = voxel_max_t.array().max(pt);
            }
            voxel_mins[thread_id] = voxel_min_t;
            voxel_maxs[thread_id] = voxel_max_t;
        }
    }
    // NaN or Inf values could exist => check for them
    else {
        // not be tested yet
#pragma omp parallel num_threads(threads_)
        {
            int thread_id = omp_get_thread_num();
            Eigen::Vector4f voxel_min_t, voxel_max_t;
            voxel_min_t.setConstant(FLT_MAX); // thread private
            voxel_max_t.setConstant(-FLT_MAX);
            //            #pragma omp single
#pragma omp for
            for (size_t i = 0; i < indices.size(); i++) {
                // Check if the point is invalid
                if (!pcl_isfinite(cloud.points[indices[i]].x) ||
                    !pcl_isfinite(cloud.points[indices[i]].y) ||
                    !pcl_isfinite(cloud.points[indices[i]].z))
                    continue;
                pcl::Array4fMapConst pt = cloud.points[indices[i]].getArray4fMap();
                voxel_min_t = voxel_min_t.array().min(pt);
                voxel_max_t = voxel_max_t.array().max(pt);
            }

            voxel_mins[thread_id] = voxel_min_t;
            voxel_maxs[thread_id] = voxel_max_t;
        }
    }

    min_pt = voxel_mins[0];
    max_pt = voxel_maxs[0];
    for (size_t i = 1; i < threads_; ++i) {
        min_pt = min_pt.array().min(voxel_mins[i].array());
        max_pt = max_pt.array().max(voxel_maxs[i].array());
    }
    //    printf("multi thread 4f cost: %fms\n", t_para.toc());
    //    printf("min max:\n");
    //    std::cout << min_pt.transpose() << std::endl;
    //    std::cout << max_pt.transpose() << std::endl;
}