#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
	if(argc != 2){
		cout << "Use: " << argv[0] << " movie" << endl;
		return -1;
	}

	const string src_movie_name = argv[1];

	VideoCapture src_movie(src_movie_name);
	if(!src_movie.isOpened()){
		cout << "can't find " << src_movie_name << endl;
		return -1;
	}

	Mat template_single = imread("template.png", IMREAD_GRAYSCALE);
	if(!template_single.data){
		cout << "can't find template.png" << endl;
		return -1;
	}

	// save to movie
	// VideoWriter dst_movie("out.mp4", VideoWriter::fourcc('X','2','6','4'), 30.0, Size(1280,720));

	namedWindow("hoge");
	uint32_t num_frame = 0;
	Mat frame_base;
	Rect area_capture(0, 200, 1280, 200);
	
	while(1){
		Mat frame;
		src_movie >> frame;
		Mat frame_broadcast = frame.clone();
		if(frame.empty() || waitKey(30) >= 0 || src_movie.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
			break;
		}
		if(num_frame == 513) frame_base = frame.clone();
		if(num_frame > 513){
			Mat frame_diff;
			absdiff(frame, frame_base, frame_diff);
			// 上から100pxを黒に
			rectangle(frame, cv::Point(0,0), cv::Point(1279, 199), cv::Scalar(0,0,0), -1, 0);
			rectangle(frame, cv::Point(0,400), cv::Point(1279, 719), cv::Scalar(0,0,0), -1, 0);
			// 差分がある場合は元画像を表示
			for(uint32_t i=0; i<1280; ++i){
				for(uint32_t j=200; j<400; ++j){
					if(frame_diff.at<Vec3b>(j,i)[0] < 10 && frame_diff.at<Vec3b>(j,i)[1] < 10 && frame_diff.at<Vec3b>(j,i)[2] < 10) frame.at<Vec3b>(j,i) = Vec3b(0,0,0);
				}
			}
			putText(frame, to_string(num_frame), Point(10,40), FONT_HERSHEY_TRIPLEX, 1.5, Scalar(100,100,250), 2, CV_AA);
			// template matching
			Mat result_img;
			Mat cropped_img(frame, area_capture);
			Mat frame_gray;
			cvtColor(cropped_img, frame_gray, CV_RGB2GRAY);
			matchTemplate(frame_gray, template_single, result_img, CV_TM_CCOEFF_NORMED);
			Rect roi_rect(0, 0, template_single.cols, template_single.rows*3);
			for(int i=0; i<10; i++){
				// 最大のスコアの場所を探す
				double maxVal;
				Point max_pt;
				minMaxLoc(result_img, NULL, &maxVal, NULL, &max_pt);
				for(int k=max_pt.y-2; k<max_pt.y+3; ++k){
					for(int l=max_pt.x-2; l<max_pt.x+3; ++l){
						if(k >= 0 && k < result_img.size().height && l >= 0 && l < result_img.size().width)
							result_img.at<float>(k, l) = 0.0f;
					}
				}
				// 一定スコア以下の場合は処理終了
				if(maxVal < 0.75) break;
 				roi_rect.x = max_pt.x;
				roi_rect.y = max_pt.y + 200;
				std::cout << "(" << max_pt.x << ", " << max_pt.y << "), score=" << maxVal << std::endl;
				// 探索結果の場所に矩形を描画
				cv::rectangle(frame, roi_rect, cv::Scalar(0,255,255), 3);
				cv::rectangle(frame_broadcast, roi_rect, cv::Scalar(0,255,255), 3);
			}
			// show
			imshow("hoge", frame);
		} else {
			putText(frame, to_string(num_frame), Point(10,40), FONT_HERSHEY_TRIPLEX, 1.5, Scalar(100,100,250), 2, CV_AA);
			imshow("hoge", frame);
		}
		// save to movie
		// dst_movie << frame_broadcast;
		
		++ num_frame;
	}

	// save to movie
	// dst_movie.release();
	
	return 0;
}
