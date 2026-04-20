//STD
#include <iostream>
#include <unistd.h>

#include <opencv2/photo.hpp>

//LIBPLENO
#include <pleno/types.h>

#include <pleno/io/printer.h>
#include <pleno/io/choice.h>

//geometry
#include <pleno/geometry/observation.h>
#include <pleno/geometry/depth/pointcloud.h>

//processing
#include <pleno/processing/detection/detection.h> 
#include <pleno/processing/imgproc/improcess.h> //devignetting, erode, dilate

#include <pleno/processing/calibration/calibration.h> 
#include <pleno/processing/depth/depth.h>
#include <pleno/processing/depth/initialization.h>
#include <pleno/processing/depth/filter.h>

#include <pleno/processing/tools/chrono.h>

//graphic
#include <pleno/graphic/display.h>

//config
#include <pleno/io/cfg/images.h>
#include <pleno/io/cfg/camera.h>
#include <pleno/io/cfg/scene.h>
#include <pleno/io/cfg/observations.h>
#include <pleno/io/cfg/poses.h>
#include <pleno/io/cfg/depths.h>

#include <pleno/io/images.h>
#include <pleno/io/depths.h>

#include "utils.h"

int main(int argc, char* argv[])
{
	PRINT_INFO("========= Lidar-Camera Calibration with a Multifocus plenoptic camera =========");
	Config_t config = parse_args(argc, argv);
	
	Viewer::enable(config.use_gui); DEBUG_VAR(Viewer::enable());
	
	Printer::verbose(config.verbose); DEBUG_VAR(Printer::verbose());
	Printer::level(config.level); DEBUG_VAR(Printer::level());

////////////////////////////////////////////////////////////////////////////////
// 1) Load Image from configuration file
////////////////////////////////////////////////////////////////////////////////
	ImageWithInfo image;
	Image mask;
	std::size_t imgformat = 8;
	{
		PRINT_WARN("1) Load Images from configuration file");
		ImagesConfig cfg_images;
		v::load(config.path.images, cfg_images);
		DEBUG_ASSERT((cfg_images.meta().rgb()), "Images must be in rgb format.");
		DEBUG_ASSERT((cfg_images.meta().format() < 16), "Floating-point images not supported.");
		
		imgformat = cfg_images.meta().format();
	
		//1.2) Load checkerboard images
		PRINT_WARN("\t1.1) Load image");	
		load(cfg_images.images()[0], image, cfg_images.meta().debayered());
		const double cbfnbr = image.fnumber;
		//1.3) Load white image corresponding to the aperture (mask)
		PRINT_WARN("\t1.2) Load white image corresponding to the aperture (mask)");
		ImageWithInfo mask_;
		load(cfg_images.mask(), mask_, cfg_images.meta().debayered());
		
		const auto [mimg, mfnbr, __] = mask_;
		mask = mimg;
		DEBUG_ASSERT((mfnbr == image.fnumber), "No corresponding f-number between mask and images");
	}
	
	PRINT_WARN("\t1.3) Devignetting images");
	Image picture = [&]() -> Image {
		Image unvignetted;
			
		if (imgformat == 8u) devignetting(image.img, mask, unvignetted);
		else /* if (imgformat == 16u) */ devignetting_u16(image.img, mask, unvignetted);
				
		return unvignetted;	
	}();
	
	Image gray = Image::zeros(picture.rows, picture.cols, CV_8UC1);
	cv::cvtColor(picture, gray, cv::COLOR_BGR2GRAY);

////////////////////////////////////////////////////////////////////////////////
// 2) Load Camera information configuration file
////////////////////////////////////////////////////////////////////////////////
	PRINT_WARN("2) Load Camera information from configuration file");
	PlenopticCamera mfpc;
	load(config.path.camera, mfpc);
	
	InternalParameters params;
	v::load(config.path.params, v::make_serializable(&params));
	mfpc.params() = params;

	PRINT_INFO("Camera = " << mfpc << std::endl);
	PRINT_INFO("Internal Parameters = " << params << std::endl);

////////////////////////////////////////////////////////////////////////////////
// 3) Load scene information
////////////////////////////////////////////////////////////////////////////////	
	PRINT_WARN("3) Load Scene Model");
	SceneConfig cfg_scene;
	v::load(config.path.scene, cfg_scene);
	DEBUG_ASSERT(
		(cfg_scene.constellations().size() > 0u),
		"No constellation available while loading scene"
	);
	
	PointsConstellation scene{cfg_scene.constellations()[0]};
	for (const auto& p : scene.constellation) DEBUG_VAR(p.transpose());

////////////////////////////////////////////////////////////////////////////////

// type == 0 默认使用光场图像计算雷达-相机标定外参   add by xyy
    if(config.type == 0)
    {
        // 4) Load bap features
    ////////////////////////////////////////////////////////////////////////////////
        BAPObservations bap_obs;
        if (config.path.features == "")
        {
            PRINT_WARN("4) Detect Features");

            //for each point in the constellation
            for (std::size_t i = 0; i < scene.size(); ++i)
            {
                PRINT_INFO("=== Detecting BAP Observation in image for point (" << i << ") in constellation");
                BAPObservations bapf = detection_bapfeatures(picture, mfpc.mia(), mfpc.params());
                DEBUG_VAR(bapf.size());

                //assign frame index
                std::for_each(bapf.begin(), bapf.end(), [&i](BAPObservation& cbo) { cbo.frame = 0; cbo.cluster = i; });

                //update observations
                bap_obs.insert(std::end(bap_obs),
                               std::make_move_iterator(std::begin(bapf)),
                               std::make_move_iterator(std::end(bapf))
                );
                DEBUG_VAR(bap_obs.size());
            }

            bap_obs.shrink_to_fit();

            ObservationsConfig cfg_obs;
            cfg_obs.features() = bap_obs;
            cfg_obs.centers() = MICObservations{};

            v::save("observations-"+std::to_string(getpid())+".bin.gz", cfg_obs);

            DEBUG_VAR(cfg_obs.features().size());
        }
        else
        {
            PRINT_WARN("4) Load Features");
            ObservationsConfig cfg_obs;
            v::load(config.path.features, cfg_obs);
            DEBUG_VAR(cfg_obs.features().size());
            bap_obs = std::move(cfg_obs.features());
            DEBUG_VAR(bap_obs.size());
        }

    ////////////////////////////////////////////////////////////////////////////////

    // 5) Optimize
    ////////////////////////////////////////////////////////////////////////////////
        PRINT_WARN("5) Calibration lidar-camera");
        PRINT_WARN("\t5.1) Load initial pose");
        CalibrationPoseConfig cfg_pose;
        v::load(config.path.extrinsics, cfg_pose);

        CalibrationPose pose{cfg_pose.pose(), cfg_pose.frame()};
        DEBUG_VAR(pose.pose);

        PRINT_WARN("\t5.2) Run calibration");
        calibration_LidarPlenopticCamera(pose, mfpc, scene, bap_obs, picture);


        // 6) PointCloud computation
////////////////////////////////////////////////////////////////////////////////s
        if (config.path.pc == "")
        {
            PRINT_WARN("6) Computing PointCloud");
            DepthMap dm;

            if (config.path.dm == "")
            {
                PRINT_WARN("\t6.1) Load depth estimation config");
                DepthEstimationStrategy strategies;
                v::load(config.path.strategy, v::make_serializable(&strategies));

                PRINT_WARN("\t6.2) Estimate depthmaps");
                const auto [mind, maxd] = initialize_min_max_distance(mfpc);
                const double dmin = strategies.dtype == DepthMap::DepthType::VIRTUAL ?
                                    strategies.vmin /* mfpc.obj2v(maxd) */
                               : 	std::max(mfpc.v2obj(strategies.vmax), mind);

                const double dmax = strategies.dtype == DepthMap::DepthType::VIRTUAL ?
                                    strategies.vmax /* mfpc.obj2v(mind) */
                                : 	std::min(mfpc.v2obj(strategies.vmin), maxd);

                const std::size_t W = strategies.mtype == DepthMap::MapType::COARSE ?
                                      mfpc.mia().width()
                                 : 	mfpc.sensor().width();

                const std::size_t H = strategies.mtype == DepthMap::MapType::COARSE ?
                                      mfpc.mia().height()
                                  : 	mfpc.sensor().height();

                DepthMap tdm{
                        W, H, dmin, dmax,
                        strategies.dtype, strategies.mtype
                };

                estimate_depth(tdm, mfpc, gray, strategies, picture, false);
                v::save("dm-"+std::to_string(getpid())+".bin.gz", v::make_serializable(&tdm));
                clear();

                config.path.dm = "./dm-"+std::to_string(getpid())+".bin.gz";
            }

            v::load(config.path.dm, v::make_serializable(&dm));
         //   inplace_minmax_filter_depth(dm, mfpc.obj2v(1500.), mfpc.obj2v(400.)); //FIXME     // by xyy: R12
            inplace_minmax_filter_depth(dm, mfpc.obj2v(8000.), mfpc.obj2v(6000.)); //FIXME     // by xyy: R2600-903

            //FIXME: filter should be applied on metric dm, as all virtual depth hypotheses are in the same unit

            PointCloud pc = [&]() -> PointCloud {
                DepthMap mdm = dm.to_metric(mfpc);

                //if (mdm.is_refined_map()) inplace_median_filter_depth(mdm, mfpc, AUTOMATIC_FILTER_SIZE, true);
                //inplace_median_filter_depth(mdm, mfpc, AUTOMATIC_FILTER_SIZE, false);

                //if (mdm.is_refined_map()) inplace_bilateral_filter_depth(mdm, mfpc, 10., 1., true);
                //inplace_bilateral_filter_depth(mdm, mfpc, 10., AUTOMATIC_FILTER_SIZE, false);

                //inplace_consistency_filter_depth(mdm, mfpc, 10. /* mm */);

             //   inplace_minmax_filter_depth(mdm, 400., 1500.); //FIXME  // by xyy: R12
                inplace_minmax_filter_depth(mdm, 6000., 8000.); //FIXME     // by xyy: R2600-903
                display(mdm, mfpc);

                return PointCloud{mdm, mfpc, picture};
            }();
            v::save("pc-"+std::to_string(getpid())+".bin.gz", v::make_serializable(&pc));

            config.path.pc = std::string("pc-"+std::to_string(getpid())+".bin.gz");
            wait();
        }
        else
        {
            PRINT_WARN("6) Load pointcloud");
        }

        FORCE_GUI(true);
        constexpr std::size_t maxcount = 500'000; //FIXME

        PointCloud pc;
        v::load(config.path.pc, v::make_serializable(&pc));
      //  inplace_minmax_filter_depth(pc, 400., 1500., Axis::Z); //FIXME    // by xyy: R12
        inplace_minmax_filter_depth(pc, 6000., 8000., Axis::Z); //FIXME      // by xyy: R2600-903
        inplace_maxcount_filter_depth(pc, maxcount);
        DEBUG_VAR(pc.size());

        PRINT_WARN("7) Graphically checking point cloud transform");
        const CalibrationPose porigin{Pose{}, -1};
        display(porigin); display(scene); display(pose);

        display(1, pc);

        PointsConstellation initial_constellation, final_constellation;
        for	(const P3D& pc : scene)
        {
            const P3D p = to_coordinate_system_of(cfg_pose.pose(), pc);
            initial_constellation.add(p);

            const P3D q = to_coordinate_system_of(pose.pose, pc);
            final_constellation.add(q);
        }
        display(initial_constellation, 10.); //constellation transformed, coord in (0,0,0), i.e. camera frame
        display(final_constellation);

        wait();

        if (config.path.pts != "")
        {
            PointCloud reference = read_pts(config.path.pts);
            display(2, reference);

            PointCloud transformed_pc = reference;
            transformed_pc.transform(pose.pose);
        //    inplace_minmax_filter_depth(transformed_pc, 400., 1500., Axis::Z); //FIXME        // by xyy: R12
            inplace_minmax_filter_depth(transformed_pc, 6000., 8000., Axis::Z); //FIXME      // by xyy: R2600-903
            //inplace_maxcount_filter_depth(transformed_pc, maxcount);
            DEBUG_VAR(transformed_pc.size());

            display(3, transformed_pc);

            wait();
        }
        FORCE_GUI(false);

    }
    else if(config.type == 1)     // type == 1 使用全聚焦图像计算雷达-相机标定外参
    {
        // 4) Load bap features
    ////////////////////////////////////////////////////////////////////////////////
        /*BAPObservations bap_obs;
        if (config.path.features == "")
        {
            PRINT_WARN("4) Detect Features");

            //for each point in the constellation
            for (std::size_t i = 0; i < scene.size(); ++i)
            {
                PRINT_INFO("=== Detecting BAP Observation in image for point (" << i << ") in constellation");
                BAPObservations bapf = detection_bapfeatures(picture, mfpc.mia(), mfpc.params());
                DEBUG_VAR(bapf.size());

                //assign frame index
                std::for_each(bapf.begin(), bapf.end(), [&i](BAPObservation& cbo) { cbo.frame = 0; cbo.cluster = i; });

                //update observations
                bap_obs.insert(std::end(bap_obs),
                               std::make_move_iterator(std::begin(bapf)),
                               std::make_move_iterator(std::end(bapf))
                );
                DEBUG_VAR(bap_obs.size());
            }


            //
            bap_obs.shrink_to_fit();

            ObservationsConfig cfg_obs;
            cfg_obs.features() = bap_obs;
            cfg_obs.centers() = MICObservations{};

            v::save("observations-"+std::to_string(getpid())+".bin.gz", cfg_obs);

            DEBUG_VAR(cfg_obs.features().size());
        }*/

    ////////////////////////////////////////////////////////////////////////////////

    // 5) Optimize
    ////////////////////////////////////////////////////////////////////////////////
        PRINT_WARN("5) Calibration lidar-camera");
        PRINT_WARN("\t5.1) Load initial pose");
        CalibrationPoseConfig cfg_pose;
        v::load(config.path.extrinsics, cfg_pose);

        CalibrationPose pose{cfg_pose.pose(), cfg_pose.frame()};
        DEBUG_VAR(pose.pose);

        PRINT_WARN("\t5.2) Run calibration");

        BAPObservations bayerCenters_AIF;
        const double ratio = double(gray.rows) / double(gray.cols);
        const int base_size = 800;
        FORCE_GUI(true);
        GUI(
                RENDER_DEBUG_2D(
                        Viewer::context().size(base_size,base_size*ratio).layer(Viewer::layer()).name("Scene"),
                        image.img
                );
        );
        Viewer::update();
        PRINT_INFO("若输入的不是全聚焦图像，请重新输入全聚焦图像");   // add by xyy
        for (std::size_t i = 0; i < scene.size(); ++i)
        {
            volatile bool finished = false;
            P2D point;
            BAPObservation bayerCenter_AIF;
            PRINT_INFO("Click on cluster corresponding to point (" << i << ") in constellation");
            Viewer::context().on_click([&](float x, float y){
                if (x > 0. and x < mfpc.sensor().width() and y > 0. and y < mfpc.sensor().height())
                {
                    PRINT_DEBUG("Click at position ("<< x << ", "<< y << ")");
                    GUI(
                            P2D c = P2D{x,y};
                            Viewer::context().layer(Viewer::layer())
                                    .name("Selected cluster")
                                    .pen_color(v::purple).pen_width(5)
                                    .add_circle(c[0], c[1], 25.)
                                    .update();

                    );
                    Viewer::update();
                    point.x() = x;
                    point.y() = y;
                    bayerCenter_AIF.u = x;      // 获取
                    bayerCenter_AIF.v = y;
                    bayerCenter_AIF.cluster = i;
                    finished = true;
                }
                else
                {
                    PRINT_ERR("Click out of bound, unvalidated. Try again.");
                    return;
                }
            });
            while (!finished)
            {
                Viewer::update();
                std::this_thread::sleep_for(std::chrono::milliseconds(20)); // 避免CPU占满
            }

            bayerCenters_AIF.push_back(bayerCenter_AIF);
            // 清理回调
            Viewer::context().on_click([](float, float){});
        }
      //  PRINT_WARN("bayerCenters_AIF.size:("<< bayerCenters_AIF.size()<< ")");
      //  calibration_LidarPlenopticCamera(pose, mfpc, scene, bap_obs, picture);
        calibration_LidarPlenopticCamera_AIF(pose, mfpc, scene, bayerCenters_AIF);

    ////////////////////////////////////////////////////////////////////////////////
// 6)  激光雷达转换为深度图
////////////////////////////////////////////////////////////////////////////////
        /*constexpr double maxd = 1500.; //FIXME: set as parameters
        constexpr double mind = 400.; //FIXME: set as parameters            // by xyy: R12*/

        /*constexpr double maxd = 8000.; //FIXME: set as parameters
        constexpr double mind = 6000.; //FIXME: set as parameters            // by xyy: HR2600-903*/

        constexpr double maxd = 62000.; //FIXME: set as parameters
        constexpr double mind = 8000.; //FIXME: set as parameters            // by xyy: R2600-801

        /*constexpr double maxd = 330000.; //FIXME: set as parameters
        constexpr double mind = 0.; //FIXME: set as parameters            // by xyy: R2600-801*/

        PointCloud references = read_pts(config.path.pts);
        references.transform(pose.pose);
        inplace_minmax_filter_depth(references, mind, maxd, Axis::Z);
        DepthMapImage dmi = DepthMapImage{references, mfpc, mind, maxd,DepthMapImage::DepthInterpMethod::MIND /* use min instead of median */};
        Image cleaned;
        cv::medianBlur(dmi.image, cleaned, 5);

        cv::imwrite("ref-csad-vd"+std::to_string(getpid())+".png", cleaned);
        cv::imwrite("ref-csad-rd-"+std::to_string(getpid())+".tiff", dmi.depthmap);
        cv::imwrite("ref-csad-rd-"+std::to_string(getpid())+".tiff", dmi.image);
      //  cv::imwrite("ref-csad-rd-"+std::to_string(getpid())+".png", dmi.depthmap);
        /*int printed1 = 0;
        for (int r = 0; r < 20; ++r) {
            for (int c = 0; c < 20; ++c) {
                const auto& di = dmi.image.at<float>(r, c);
                std::cout << "#" << printed1
                          << " pix=(" << r << "," << c << ")"
                          << " csad_depth=" << di
                          << "\n";
                ++printed1;
            }
        }*/

    ////////////////////////////////////////////////////////////////////////////////
// 7)  计算真实深度与估计深度间的绝对误差
////////////////////////////////////////////////////////////////////////////////
        DepthMap dm;
        v::load(config.path.dm,v::make_serializable(&dm));

        int printed2 = 0;
        for (int r = 0; r < 20; ++r) {
            for (int c = 0; c < 20; ++c) {
                const auto& di = dm.map(r, c);
                /*if (only_valid) {
                    if (di.state != DepthMap::DepthInfo::COMPUTED) continue;
                    if (di.depth == DepthMap::DepthInfo::NO_DEPTH) continue;
                }*/
                std::cout << "#" << printed2
                          << " pix=(" << r << "," << c << ")"
                          << " depth=" << di.depth
                          << " conf=" << di.confidence
                          << "\n";
                ++printed2;
            }
        }

        PointCloud pc = [&]() -> PointCloud {
                        DepthMap mdm = dm.to_metric(mfpc);
                        inplace_minmax_filter_depth(mdm, mind, maxd);
                     //   display(mdm, mfpc);
                        return PointCloud{mdm, mfpc, picture};
        }();



        constexpr std::size_t maxcount = 50'000; //FIXME
        inplace_maxcount_filter_depth(references, maxcount);
        DEBUG_VAR(references.size());

        PRINT_INFO("=== Displaying pointcloud ("<< cfg_pose.frame() <<")");
        display(cfg_pose.frame(), references);

        auto chamfer_distance = [](const PointCloud& ref, const PointCloud& reading) -> double {
            return (
                    0. + //ref.distance(reading, PointCloud::DistanceType::Chamfer)
                    + reading.distance(ref, PointCloud::DistanceType::Chamfer)
            );
        };

        auto hausdorff_distance = [](const PointCloud& ref, const PointCloud& reading) -> double {
            return (
                    0. + //ref.distance(reading, PointCloud::DistanceType::Hausdorff)
                    + reading.distance(ref, PointCloud::DistanceType::Hausdorff)
            );
        };

        {
            PRINT_DEBUG("=== Starting computing Chamfer distance....");
            Chrono::tic();
            const double score = chamfer_distance(references, pc);          //   references:真实深度; pc：算法估计深度
            Chrono::tac();
            PRINT_DEBUG("...Finished!");

            PRINT_INFO("(frame = "<< cfg_pose.frame() << ") D(ref, reading) = "<< score << " (computed in "<< Chrono::get() << " s)!");
        }
        {
            PRINT_DEBUG("=== Starting computing Hausdorff distance....");
            Chrono::tic();
            const double score = hausdorff_distance(references, pc);
            Chrono::tac();
            PRINT_DEBUG("...Finished!");

            PRINT_INFO("(frame = "<< cfg_pose.frame() << ") D(ref, reading) = "<< score << " (computed in "<< Chrono::get() << " s)!");
        }
////////////////////////////////////////////////////////////////////////////////
// 8)  多项式拟合求解c0,c1,c2
////////////////////////////////////////////////////////////////////////////////

        /*std::array<double,3>;
        BehavioralModel( std::vector<double>& realdepth, std::vector<double>& virtualdepth)
        {
            CV_Assert(realdepth.size() == virtualdepth.size());
            const int N = static_cast<int>(realdepth.size());
            CV_Assert(N >= 3); // 至少 3 个点才能估 3 个参数

            cv::Mat X(N, 3, CV_64F); // [u, v, 1]
            cv::Mat y(N, 1, CV_64F); // a_L

            for (int i = 0; i < N; ++i) {
                const double aL = realdepth[i];
                const double v  = virtualdepth[i];
                const double u  = aL * v;           // 论文定义：u = a_L * v

                X.at<double>(i, 0) = u;             // 列 0: u
                X.at<double>(i, 1) = v;             // 列 1: v
                X.at<double>(i, 2) = 1.0;           // 列 2: 常数项
                y.at<double>(i, 0) = aL;            // 目标：a_L
            }

            cv::Mat c; // 3x1
            bool ok = cv::solve(X, y, c, cv::DECOMP_SVD);  // 最小二乘（稳健）
            if (!ok) {
                throw std::runtime_error("FitBehavioralModel: cv::solve failed.");
            }

            // 按顺序输出 c0, c1, c2
            return { c.at<double>(0,0), c.at<double>(1,0), c.at<double>(2,0) };
        }*/


    }



// 6) PointCloud computation
////////////////////////////////////////////////////////////////////////////////
	/*if (config.path.pc == "")
	{
		PRINT_WARN("6) Computing PointCloud");
		DepthMap dm;
		
		if (config.path.dm == "")
		{
			PRINT_WARN("\t6.1) Load depth estimation config");		
			DepthEstimationStrategy strategies;
			v::load(config.path.strategy, v::make_serializable(&strategies));
			
			PRINT_WARN("\t6.2) Estimate depthmaps");	
			const auto [mind, maxd] = initialize_min_max_distance(mfpc);
			const double dmin = strategies.dtype == DepthMap::DepthType::VIRTUAL ? 
					strategies.vmin // mfpc.obj2v(maxd)
				: 	std::max(mfpc.v2obj(strategies.vmax), mind);
			
			const double dmax = strategies.dtype == DepthMap::DepthType::VIRTUAL ? 
					strategies.vmax // mfpc.obj2v(mind)
				: 	std::min(mfpc.v2obj(strategies.vmin), maxd);
				
			const std::size_t W = strategies.mtype == DepthMap::MapType::COARSE ? 
					mfpc.mia().width() 
				: 	mfpc.sensor().width();
				
			const std::size_t H = strategies.mtype == DepthMap::MapType::COARSE ? 
					mfpc.mia().height() 
				: 	mfpc.sensor().height();
				
			DepthMap tdm{
				W, H, dmin, dmax,
				strategies.dtype, strategies.mtype
			};

			estimate_depth(tdm, mfpc, gray, strategies, picture, false);
			v::save("dm-"+std::to_string(getpid())+".bin.gz", v::make_serializable(&tdm));
			clear();
			
			config.path.dm = "./dm-"+std::to_string(getpid())+".bin.gz";
		}
		
		v::load(config.path.dm, v::make_serializable(&dm));
		inplace_minmax_filter_depth(dm, mfpc.obj2v(1500.), mfpc.obj2v(400.)); //FIXME	
		
		//FIXME: filter should be applied on metric dm, as all virtual depth hypotheses are in the same unit

		PointCloud pc = [&]() -> PointCloud {
			DepthMap mdm = dm.to_metric(mfpc);
			
			//if (mdm.is_refined_map()) inplace_median_filter_depth(mdm, mfpc, AUTOMATIC_FILTER_SIZE, true);
			//inplace_median_filter_depth(mdm, mfpc, AUTOMATIC_FILTER_SIZE, false);		
			
			//if (mdm.is_refined_map()) inplace_bilateral_filter_depth(mdm, mfpc, 10., 1., true);
			//inplace_bilateral_filter_depth(mdm, mfpc, 10., AUTOMATIC_FILTER_SIZE, false);	
			
			//inplace_consistency_filter_depth(mdm, mfpc, 10. // mm );
				
			inplace_minmax_filter_depth(mdm, 400., 1500.); //FIXME	
			display(mdm, mfpc);
		
			return PointCloud{mdm, mfpc, picture};
		}();		
		v::save("pc-"+std::to_string(getpid())+".bin.gz", v::make_serializable(&pc));
		
		config.path.pc = std::string("pc-"+std::to_string(getpid())+".bin.gz");
		wait();
	}
	else
	{
		PRINT_WARN("6) Load pointcloud");
	}
	
FORCE_GUI(true);	
	constexpr std::size_t maxcount = 500'000; //FIXME	
	
	PointCloud pc; 
	v::load(config.path.pc, v::make_serializable(&pc));
	inplace_minmax_filter_depth(pc, 400., 1500., Axis::Z); //FIXME	
	inplace_maxcount_filter_depth(pc, maxcount);
	DEBUG_VAR(pc.size());*/
	
	/*PRINT_WARN("7) Graphically checking point cloud transform");
	const CalibrationPose porigin{Pose{}, -1}; 
	display(porigin); display(scene); display(pose);
	
	display(1, pc);
	
	PointsConstellation initial_constellation, final_constellation;
	for	(const P3D& pc : scene)
	{
		const P3D p = to_coordinate_system_of(cfg_pose.pose(), pc);
		initial_constellation.add(p);
		
		const P3D q = to_coordinate_system_of(pose.pose, pc);
		final_constellation.add(q);
	}
	display(initial_constellation, 10.); //constellation transformed, coord in (0,0,0), i.e. camera frame
	display(final_constellation); 
	
	wait();
	
	if (config.path.pts != "")
	{
		PointCloud reference = read_pts(config.path.pts);		
		display(2, reference);	
		
		PointCloud transformed_pc = reference;
		transformed_pc.transform(pose.pose);
		inplace_minmax_filter_depth(transformed_pc, 400., 1500., Axis::Z); //FIXME	
		//inplace_maxcount_filter_depth(transformed_pc, maxcount);
		DEBUG_VAR(transformed_pc.size());
		
		display(3, transformed_pc);			
		
		wait();	
	}*/
FORCE_GUI(false);		

	PRINT_INFO("========= EOF =========");

	Viewer::wait();
	Viewer::stop();
	return 0;
}

