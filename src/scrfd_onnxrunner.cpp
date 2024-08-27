
/**/

#include <iostream>
#include <Windows.h>

#include "../include/scrfd_onnxrunner.h"

static void test_default()
{
    std::string onnx_path = "${YourOnnxModelPath}";
    onnx_path = "../../../examples/hub/onnx/cv/scrfd_2.5g_bnkps_shape640x640.onnx";

    std::string test_img_path = "${YourTestImagePath}";
    test_img_path = "../../../examples/lite/resources/test_lite_face_detector.jpg";

    std::string save_img_path = "${YourSaveImagePath}";
    save_img_path = "../../../examples/logs/test_lite_scrfd.jpg";

    Configuration cfg;
    cfg.modelpath = onnx_path;

    SCRFDOnnxRunner *FaceDetector = new SCRFDOnnxRunner(cfg);



}

inline wchar_t* multi_Byte_To_Wide_Char(std::string& pKey)
{
    // string 转 char*
    const char* pCStrKey = pKey.c_str();
    // 第一次调用返回转换后的字符串长度，用于确认为wchar_t*开辟多大的内存空间
    size_t pSize = MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, NULL, 0);
    wchar_t* pWCStrKey = new wchar_t[pSize];
    MultiByteToWideChar(CP_OEMCP, 0, pCStrKey, strlen(pCStrKey) + 1, pWCStrKey, pSize);
    // 不要忘记在使用完wchar_t*后delete[]释放内存
    return pWCStrKey;
}

SCRFDOnnxRunner::SCRFDOnnxRunner(Configuration cfg , unsigned int threads) : \
    num_threads(threads)
{
    this->conf_threshold = cfg.conf_threshold;
    this->nms_threshold = cfg.nms_threshold;

}

SCRFDOnnxRunner::~SCRFDOnnxRunner()
{
}

float SCRFDOnnxRunner::GetConfThreshold()
{
    return this->conf_threshold;
}

void SCRFDOnnxRunner::SetConfThreshold(float thresh)
{
    this-> conf_threshold = thresh;
}

float SCRFDOnnxRunner::GetNMSThreshold()
{
    return this->nms_threshold;
}

void SCRFDOnnxRunner::SetNMSThreshold(float thresh)
{
    this-> nms_threshold = thresh;
}

int SCRFDOnnxRunner::InitOrtEnv(Configuration cfg)
{
    std::cout << "< - * -------- INITIAL ONNXRUNTIME ENV START -------- * ->" << std::endl;
    try
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SCRFDOnnxRunner");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (cfg.device == "cuda") {
            std::cout << "[INFO] OrtSessionOptions Append CUDAExecutionProvider" << std::endl;
            OrtCUDAProviderOptions cuda_options{};

            cuda_options.device_id = 0;
            cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchDefault;
            cuda_options.gpu_mem_limit = 0; 
            cuda_options.arena_extend_strategy = 1; // 设置GPU内存管理中的Arena扩展策略
            cuda_options.do_copy_in_default_stream = 1; // 是否在默认CUDA流中执行数据复制
            cuda_options.has_user_compute_stream = 0;
            cuda_options.default_memory_arena_cfg = nullptr;

            session_options.AppendExecutionProvider_CUDA(cuda_options);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        }

        #if _WIN32
            std::cout << "[INFO] Env _WIN32 change modelpath from multi byte to wide char ..." << std::endl;
            const wchar_t* modelPath = multi_Byte_To_Wide_Char(cfg.modelpath);
        #else
            const char* modelPath = cfg.lightgluePath;
        #endif // _WIN32

        session = std::make_unique<Ort::Session>(env , modelPath , session_options);

        const size_t numInputNodes = session->GetInputCount();
        InputNodeNames.reserve(numInputNodes);
        for (size_t i = 0 ; i < numInputNodes ; i++)
        {
            InputNodeNames.emplace_back(_strdup(session->GetInputNameAllocated(i , allocator).get()));
            InputNodeShapes.emplace_back(session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }

        const size_t numOutputNodes = session->GetOutputCount();
        OutputNodeNames.reserve(numOutputNodes);
        for (size_t i = 0 ; i < numOutputNodes ; i++)
        {
            OutputNodeNames.emplace_back(_strdup(session->GetOutputNameAllocated(i , allocator).get()));
            OutputNodeShapes.emplace_back(session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
        }
        
        std::cout << "[INFO] ONNXRuntime environment created successfully." << std::endl;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] ONNXRuntime environment created failed : " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}

cv::Mat SCRFDOnnxRunner::NormalizeImage(cv::Mat& Image)
{
    cv::Mat normalizedImage = Image.clone();

    if (Image.channels() == 3) {
        cv::cvtColor(normalizedImage, normalizedImage, cv::COLOR_BGR2RGB);
        normalizedImage.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
    } else if (Image.channels() == 1) {
        Image.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
    } else {
        throw std::invalid_argument("[ERROR] Not an image");
    }

    return normalizedImage;
}

cv::Mat SCRFDOnnxRunner::ResizeImage(const cv::Mat srcImage, int* newh, int* neww, int* top, int* left)
{
	cv::Mat dstimg;
    *newh = this->input_height;
	*neww = this->input_width;
    int srch = srcImage.rows, srcw = srcImage.cols;

	if (this->keep_ratio && srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = this->input_height;
			*neww = int(this->input_width / hw_scale);
			cv::resize(srcImage, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((this->input_width - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->input_width - *neww - *left, cv::BORDER_CONSTANT, 0);
		}
		else
		{
			*newh = (int)this->input_height * hw_scale;
			*neww = this->input_width;
			cv::resize(srcImage, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(this->input_height - *newh) * 0.5;
			cv::copyMakeBorder(dstimg, dstimg, *top, this->input_height - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 0);
		}
	}
	else
	{
		cv::resize(srcImage, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}

	return dstimg;
}

cv::Mat SCRFDOnnxRunner::RGB2Grayscale(cv::Mat& Image) {
    cv::Mat resultImage;
    cv::cvtColor(Image, resultImage, cv::COLOR_RGB2GRAY);

    return resultImage;
}

cv::Mat SCRFDOnnxRunner::PreProcess(Configuration cfg , const cv::Mat& Image)
{
    cv::Mat tempImage = Image.clone();
    std::cout << "[INFO] Image info :  width : " << Image.cols << " height :  " << Image.rows << std::endl;
    
    int newh = 0, neww = 0, padh = 0, padw = 0;
    cv::Mat resultImage = NormalizeImage(ResizeImage(tempImage , &newh, &neww, &padh, &padw ));
   
    return resultImage;
}

int SCRFDOnnxRunner::Inference(Configuration cfg , const cv::Mat& src)
{   
    try 
    {   
        // Dynamic InputNodeShapes is [1,3,-1,-1]  
        std::cout << "[INFO] srcImage Size : " << src.size() << " Channels : " << src.channels() << std::endl;
        
        // Build src input node shape and destImage input node shape
        int srcInputTensorSize;
        
        srcInputTensorSize = InputNodeShapes[0][0] * InputNodeShapes[0][1] * InputNodeShapes[0][2] * InputNodeShapes[0][3];
        std::vector<float> srcInputTensorValues(srcInputTensorSize);

        if (true)
        {
            srcInputTensorValues.assign(src.begin<float>() , src.end<float>());
        }else{             
            int src_height = src.rows;
            int src_width = src.cols;
            for (int y = 0; y < src_height; y++) {
                for (int x = 0; x < src_width; x++) {
                    cv::Vec3f pixel = src.at<cv::Vec3f>(y, x); // RGB
                    srcInputTensorValues[y * src_width + x] = pixel[2];
                    srcInputTensorValues[src_height * src_width + y * src_width + x] = pixel[1];
                    srcInputTensorValues[2 * src_height * src_width + y * src_width + x] = pixel[0];
                }
            }
        }

        auto memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
        
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_handler , srcInputTensorValues.data() , srcInputTensorValues.size() , \
            InputNodeShapes[0].data() , InputNodeShapes[0].size()
        ));

        auto time_start = std::chrono::high_resolution_clock::now();

        auto output_tensor = session->Run(Ort::RunOptions{nullptr} , InputNodeNames.data() , input_tensors.data() , \
                    input_tensors.size() , OutputNodeNames.data() , OutputNodeNames.size());
        
        auto time_end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();

        for (auto& tensor : output_tensor)
        {
            if (!tensor.IsTensor() || !tensor.HasValue())
            {
                std::cerr << "[ERROR] Inference output tensor is not a tensor or don't have value" << std::endl;
            }
        }
        output_tensors = std::move(output_tensor);

        std::cout << "[INFO] SCRFDOnnxRunner inference finish ..." << std::endl;
	    std::cout << "[INFO] Inference cost time : " << diff << "ms" << std::endl;
    } 
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] SCRFDOnnxRunner inference failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}


// void SCRFD::generate_bboxes_single_stride(Ort::Value &score_pred, Ort::Value &bbox_pred,
//     unsigned int stride, float score_threshold, float img_height, float img_width,
//     std::vector<types::BoxfWithLandmarks> &bbox_kps_collection)
// {
//   unsigned int nms_pre_ = (stride / 8) * nms_pre; // 1 * 1000,2*1000,...
//   nms_pre_ = nms_pre_ >= nms_pre ? nms_pre_ : nms_pre;

//   auto stride_dims = score_pred.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape();
//   const unsigned int num_points = stride_dims.at(1);  // 12800
//   const float *score_ptr = score_pred.GetTensorMutableData<float>(); // [1,12800,1]
//   const float *bbox_ptr = bbox_pred.GetTensorMutableData<float>();   // [1,12800,4]

//   float ratio = scale_params.ratio;
//   int dw = scale_params.dw;
//   int dh = scale_params.dh;

//   unsigned int count = 0;
//   auto &stride_points = center_points[stride];

//   for (unsigned int i = 0; i < num_points; ++i)
//   {
//     const float cls_conf = score_ptr[i];
//     if (cls_conf < score_threshold) continue; // filter
//     auto &point = stride_points.at(i);
//     const float cx = point.cx; // cx
//     const float cy = point.cy; // cy
//     const float s = point.stride; // stride

//     // bbox
//     const float *offsets = bbox_ptr + i * 4;
//     float l = offsets[0]; // left
//     float t = offsets[1]; // top
//     float r = offsets[2]; // right
//     float b = offsets[3]; // bottom

//     types::BoxfWithLandmarks box_kps;
//     float x1 = ((cx - l) * s - (float) dw) / ratio;  // cx - l x1
//     float y1 = ((cy - t) * s - (float) dh) / ratio;  // cy - t y1
//     float x2 = ((cx + r) * s - (float) dw) / ratio;  // cx + r x2
//     float y2 = ((cy + b) * s - (float) dh) / ratio;  // cy + b y2
//     box_kps.box.x1 = std::max(0.f, x1);
//     box_kps.box.y1 = std::max(0.f, y1);
//     box_kps.box.x2 = std::min(img_width - 1.f, x2);
//     box_kps.box.y2 = std::min(img_height - 1.f, y2);
//     box_kps.box.score = cls_conf;
//     box_kps.box.label = 1;
//     box_kps.box.label_text = "face";
//     box_kps.box.flag = true;
//     box_kps.flag = true;

//     bbox_kps_collection.push_back(box_kps);

//     count += 1; // limit boxes for nms.
//     if (count > max_nms)
//       break;
//   }

//   if (bbox_kps_collection.size() > nms_pre_)
//   {
//     std::sort(
//         bbox_kps_collection.begin(), bbox_kps_collection.end(),
//         [](const types::BoxfWithLandmarks &a, const types::BoxfWithLandmarks &b)
//         { return a.box.score > b.box.score; }
//     ); // sort inplace
//     // trunc
//     bbox_kps_collection.resize(nms_pre_);
//   }

// }

int SCRFDOnnxRunner::PostProcess(Configuration cfg)
{
    try{

        // score_8,score_16,score_32,bbox_8,bbox_16,bbox_32
        Ort::Value &score_8 = output_tensors.at(0);  // e.g [1,12800,1]
        Ort::Value &score_16 = output_tensors.at(1); // e.g [1,3200,1]
        Ort::Value &score_32 = output_tensors.at(2); // e.g [1,800,1]
        Ort::Value &bbox_8 = output_tensors.at(3);   // e.g [1,12800,4]
        Ort::Value &bbox_16 = output_tensors.at(4);  // e.g [1,3200,4]
        Ort::Value &bbox_32 = output_tensors.at(5);  // e.g [1,800,4]
        // // generate center points.
        // const float input_height = static_cast<float>(input_node_dims.at(2)); // e.g 640
        // const float input_width = static_cast<float>(input_node_dims.at(3)); // e.g 640
        // this->generate_points(input_height, input_width);

        // bbox_kps_collection.clear();

        // // level 8 & 16 & 32
        // this->generate_bboxes_single_stride(scale_params, score_8, bbox_8, 8, score_threshold,
        //                                     img_height, img_width, bbox_kps_collection);
        // this->generate_bboxes_single_stride(scale_params, score_16, bbox_16, 16, score_threshold,
        //                                     img_height, img_width, bbox_kps_collection);
        // this->generate_bboxes_single_stride(scale_params, score_32, bbox_32, 32, score_threshold,
        //                                     img_height, img_width, bbox_kps_collection);
        
    }
    catch(const std::exception& ex)
    {
        std::cerr << "[ERROR] PostProcess failed : " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


std::vector<FaceObject> SCRFDOnnxRunner::InferenceImage(Configuration cfg , const cv::Mat& srcImage)
{   
    std::cout << "< - * -------- INFERENCEIMAGE START -------- * ->" << std::endl;

    if (srcImage.empty())
	{
		throw  "[ERROR] ImageEmptyError ";
	}
    cv::Mat srcImage_copy = cv::Mat(srcImage);
    cv::Mat src = PreProcess(cfg , srcImage_copy);
    
    Inference(cfg , src);

    PostProcess(cfg);

    output_tensors.clear();

}

void SCRFDOnnxRunner::Draw(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];

        fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));

        char text[256];
        sprintf(text, "%.1f%%", obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::imshow("image", image);
    cv::waitKey(0);
}



