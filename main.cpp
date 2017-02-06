#include <iostream>
#include <string>
#include <vector>
#include <iterator>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

using namespace std;
using namespace cv;

static int matching_height = 20; // パターンマッチングを行う領域の高さ(A)

static float matching_score = 0.65; // パターンマッチ成功とみなす最小値

// template matching
vector<Rect2d> testPattern(Mat& frame, Mat& frame_broadcast, Rect& area_capture, Mat& template_single){
	Mat result_img;
	Mat cropped_img(frame, area_capture);
	Mat frame_gray;
	cvtColor(cropped_img, frame_gray, CV_RGB2GRAY);
	Mat cropped_hsv;
	cvtColor(cropped_img, cropped_hsv, CV_RGB2HSV);
	matchTemplate(frame_gray, template_single, result_img, CV_TM_CCOEFF_NORMED);
	Rect roi_rect(0, 0, template_single.cols, template_single.rows*3);
	vector<Rect2d> bbox;
	for(int i=0; i<10; i++){
		// 最大のスコアの場所を探す
		double maxVal;
		Point max_pt;
		minMaxLoc(result_img, NULL, &maxVal, NULL, &max_pt);
		for(int k=max_pt.y-10; k<max_pt.y+11; ++k){
			for(int l=max_pt.x-10; l<max_pt.x+11; ++l){
				if(k >= 0 && k < result_img.size().height && l >= 0 && l < result_img.size().width)
					result_img.at<float>(k, l) = 0.0f;
			}
		}
		// 一定スコア以下の場合は処理終了
		if(maxVal < matching_score) break;
		roi_rect.x = max_pt.x;
		roi_rect.y = max_pt.y + area_capture.y;
		cout << "(" << max_pt.x << ", " << max_pt.y << "), score=" << maxVal;
		// 返り値に追加
		bbox.push_back(Rect2d(roi_rect.x, roi_rect.y, template_single.cols, template_single.rows));
		// 探索結果の場所に矩形を描画
		Scalar roicolor(0, 0, 0);
		unsigned char pupcolor = cropped_hsv.at<Vec3b>(max_pt.y+(template_single.rows/2), max_pt.x+(template_single.cols/2))[0];
		if(pupcolor >= 110 && pupcolor < 150){
			roicolor[2] = 255;
			cout << ", color=red" << endl;
			cv::rectangle(frame, roi_rect, roicolor, 3);
			cv::rectangle(frame_broadcast, roi_rect, roicolor, 3);
		} else if(pupcolor >= 90 && pupcolor < 110){
			roicolor[2] = 255;
			roicolor[1] = 130;
			cout << ", color=orange" << endl;
			cv::rectangle(frame, roi_rect, roicolor, 3);
			cv::rectangle(frame_broadcast, roi_rect, roicolor, 3);
		} else {
			roicolor[1] = 255;
			cout << ", color=green";
			int16_t flag_l = 0;
			int16_t flag_r = 0;
			cout << endl;
			for(int i=template_single.rows/2; i<template_single.rows*5; i+=1){
				cout << (int)(cropped_hsv.at<Vec3b>(max_pt.y+i, max_pt.x+(template_single.cols/2)-5)[1]) << ", " << (int)(cropped_hsv.at<Vec3b>(max_pt.y+i, max_pt.x+(template_single.cols/2)+5)[1]) << endl;
				if(flag_l==0 && cropped_hsv.at<Vec3b>(max_pt.y+i, max_pt.x+(template_single.cols/2)+5)[1] > 100){
					flag_l = -1;
				} else if(flag_l == -1 && cropped_hsv.at<Vec3b>(max_pt.y+i, max_pt.x+(template_single.cols/2)+5)[1] < 100){
					flag_l = 1;
					break;
				}
				if(flag_r==0 && cropped_hsv.at<Vec3b>(max_pt.y+i, max_pt.x+(template_single.cols/2)-5)[1] > 100){
					flag_r = -1;
				} else if(flag_r == -1 && cropped_hsv.at<Vec3b>(max_pt.y+i, max_pt.x+(template_single.cols/2)-5)[1] < 100){
					flag_r = 1;
					break;
				}
			}
			if(flag_l == 1){
				cout << ":left" << endl;
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y+3/2*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y), roicolor, 3);
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y+3/2*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
				cv::line(frame, Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y+3/2*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y+3/2*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
			} else {
				cout << ":right" << endl;
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y+3*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y+3*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
			}
		}
	}
	return bbox;
}

int main(int argc, char** argv){
	if(argc != 2 && argc != 3){
		cout << "Use: " << argv[0] << " filein [fileout]" << endl;
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

	bool is_store = false;
	VideoWriter dst_movie;
	if(argc == 3){
		// save to movie
		dst_movie.open(argv[2], VideoWriter::fourcc('X','2','6','4'), 30.0, Size(1280,720));
		is_store = true;
	}

	namedWindow("hoge");
	uint32_t num_frame = 0;
	Mat frame_base;
	Rect area_capture_a(0, 200, 1280, matching_height);
	Rect area_capture_b(0, 300, 1280, matching_height);

	vector< Ptr<Tracker> > tracker;
	vector<Rect2d> tracked_bbox;

	while(1){
		Mat frame;
		src_movie >> frame;
		Mat frame_broadcast = frame.clone();
		if(frame.empty() || waitKey(30) >= 0 || src_movie.get(CV_CAP_PROP_POS_AVI_RATIO) == 1){
			break;
		}
		if(num_frame == 513){
			frame_base = frame.clone();
		} else if(num_frame > 513){
			Mat frame_diff;
			absdiff(frame, frame_base, frame_diff);
			// // 上から100pxを黒に
			// rectangle(frame, cv::Point(0,0), cv::Point(1279, 199), cv::Scalar(0,0,0), -1, 0);
			// rectangle(frame, cv::Point(0,199+matching_height), cv::Point(1279, 299), cv::Scalar(0,0,0), -1, 0);
			// rectangle(frame, cv::Point(0,299+matching_height), cv::Point(1279, 719), cv::Scalar(0,0,0), -1, 0);
			// 差分がある場合は元画像を表示
			for(uint32_t i=0; i<1280; ++i){
				for(uint32_t j=0; j<720; ++j){
					if(frame_diff.at<Vec3b>(j,i)[0] < 10 && frame_diff.at<Vec3b>(j,i)[1] < 10 && frame_diff.at<Vec3b>(j,i)[2] < 10) frame.at<Vec3b>(j,i) = Vec3b(0,0,0);
				}
			}
			putText(frame, to_string(num_frame), Point(10,40), FONT_HERSHEY_TRIPLEX, 1.5, Scalar(100,100,250), 2, CV_AA);
			putText(frame_broadcast, to_string(num_frame), Point(10,40), FONT_HERSHEY_TRIPLEX, 1.5, Scalar(100,100,250), 2, CV_AA);

			vector<Rect2d> bboxes = testPattern(frame, frame_broadcast, area_capture_a, template_single);

			if(!bboxes.empty()){
				Ptr<Tracker> tmp = Tracker::create("KCF");
				tmp->init(frame, bboxes.at(0));
				tracker.push_back(tmp);
			}

			tracked_bbox.clear();
			for(auto ite = tracker.begin(); ite != tracker.end(); ++ite){
				Rect2d tmp_bbox;
				tmp_bbox.width = template_single.cols;
				tmp_bbox.height = template_single.rows;
				(*ite)->update(frame, tmp_bbox);
				
				if(tmp_bbox.y > 400) tracker.erase(ite);
				else tracked_bbox.push_back(tmp_bbox);
			}
			
			cout << "* tracked bbox" << endl;
			for(auto i : tracked_bbox){
				rectangle(frame, i, Scalar(0,0,255), 3);
				cout << "x:" << (int)(i.x) << ", y" << (int)(i.y) << ", w" << (int)(i.width) << ", h" << (int)(i.height) << endl;
			}

			// show
			imshow("hoge", frame);
		} else {
			putText(frame, to_string(num_frame), Point(10,40), FONT_HERSHEY_TRIPLEX, 1.5, Scalar(100,100,250), 2, CV_AA);
			imshow("hoge", frame);
		}
		// save to movie
		if(is_store) dst_movie << frame_broadcast;
		
		++ num_frame;
	}

	// save to movie
	if(is_store) dst_movie.release();
	
	return 0;
}
