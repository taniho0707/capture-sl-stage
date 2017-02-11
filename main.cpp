#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <iterator>
#include <algorithm>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

using namespace std;
using namespace cv;

enum class notetype : unsigned char {
	SINGLE,
	LONG_START,
	LONG_END,
	SLIDERIGHT_START,
	SLIDERIGHT_CONT,
	SLIDERIGHT_END,
	SLIDELEFT_START,
	SLIDELEFT_CONT,
	SLIDELEFT_END,
	ERROR = 15
};

enum class noteline : uint8_t {
	LEFT,
	LEFTMIDDLE,
	MIDDLE,
	RIGHTMIDDLE,
	RIGHT,
	NONE = 7
};


typedef pair<notetype, Rect2d> TrackedNote_t;

static int matching_height = 20; // パターンマッチングを行う領域の高さ(A)
static float matching_score = 0.65; // パターンマッチ成功とみなす最小値
static int tappoint_y = 598; // タップするラインのy座標

static int estimate_time = 10; // 最小二乗法に使うポイントの数

static array< pair<int, int>, 5 > line_boundary{
	pair<int,int>{0, 250},
	pair<int,int>{251, 480},
	pair<int,int>{481, 740},
	pair<int,int>{741, 950},
	pair<int,int>{951, 1279}
};

static array<int, 5> line_pos{ 215, 431, 640, 856, 1066 };


Scalar getNotesColor(notetype type){
	if(type == notetype::SINGLE){
		return Scalar(0, 0, 255);
	} else if(type == notetype::LONG_START || type == notetype::LONG_END){
		return Scalar(0, 130, 255);
	} else if(type == notetype::ERROR){
		return Scalar(255, 255, 255);
	} else {
		return Scalar(0, 255, 0);
	}
}

int calcTappointX(Rect2d pos1, Rect2d pos2){
	// |    y = ax + b
	// |
	// -----> y
	// |
	// \/
	// x
	double a = (double)(pos2.x-pos1.x)/(pos2.y-pos1.y);
	double b = (double)(pos1.x) - a*pos1.y;
	return (int)(a*tappoint_y + b);
}

int calcLSM(vector<Point> p){
	Vec4f dst;
	fitLine(Mat(p), dst, CV_DIST_L1, 0, 0.01, 0.01);
	return static_cast<int>(((tappoint_y-dst[2])*dst[1]/dst[0])+dst[3]);
}

int adjustEstimatedPos(int real){
	int estimated = 255; // big number
	for(int i=0; i<5; ++i){
		if(real >= line_boundary.at(i).first && real < line_boundary.at(i).second){
			estimated = i;
			break;
		}
	}
	if(estimated == 255) return 0;
	else return line_pos.at(estimated);
}

int getNonZeroRows(Mat& mat){
	for(int i=0; i<estimate_time; ++i){
		if(mat.at<uint64_t>(i, 0) == 0){
			return i;
		}
	}
	return estimate_time;
}

// template matching
vector<TrackedNote_t> testPattern(Mat& frame, Mat& frame_broadcast, Rect& area_capture, Mat& template_single){
	Mat result_img;
	Mat cropped_img(frame, area_capture);
	Mat frame_gray;
	cvtColor(cropped_img, frame_gray, CV_RGB2GRAY);
	Mat cropped_hsv;
	cvtColor(cropped_img, cropped_hsv, CV_RGB2HSV);
	matchTemplate(frame_gray, template_single, result_img, CV_TM_CCOEFF_NORMED);
	Rect roi_rect(0, 0, template_single.cols, template_single.rows*3);
	vector<TrackedNote_t> bbox;
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
		// 探索結果の場所に矩形を描画
		unsigned char pupcolor = cropped_hsv.at<Vec3b>(max_pt.y+(template_single.rows/2), max_pt.x+(template_single.cols/2))[0];
		notetype type;
		if(pupcolor >= 110 && pupcolor < 150){
			Scalar roicolor = getNotesColor(notetype::SINGLE);
			cout << ", color=red" << endl;
			cv::rectangle(frame, roi_rect, roicolor, 3);
			cv::rectangle(frame_broadcast, roi_rect, roicolor, 3);
			type = notetype::SINGLE;
		} else if(pupcolor >= 90 && pupcolor < 110){
			Scalar roicolor = getNotesColor(notetype::LONG_START);
			cout << ", color=orange" << endl;
			cv::rectangle(frame, roi_rect, roicolor, 3);
			cv::rectangle(frame_broadcast, roi_rect, roicolor, 3);
			type = notetype::LONG_START;
		} else {
			Scalar roicolor = getNotesColor(notetype::SLIDELEFT_CONT);
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
				type = notetype::SLIDELEFT_CONT;
			} else {
				cout << ":right" << endl;
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y+3*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y+3*template_single.rows), Point(max_pt.x+template_single.cols, max_pt.y+area_capture.y+template_single.rows/2*3), roicolor, 3);
				cv::line(frame_broadcast, Point(max_pt.x, max_pt.y+area_capture.y), Point(max_pt.x, max_pt.y+area_capture.y+template_single.rows*3), roicolor, 3);
				type = notetype::SLIDERIGHT_CONT;
			}
		}
		// 返り値に追加
		bbox.push_back(TrackedNote_t(type, Rect2d(roi_rect.x, roi_rect.y, template_single.cols, template_single.rows)));
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
	// Rect area_capture_b(0, 300, 1280, matching_height);

	vector< Ptr<Tracker> > tracker;
	vector<TrackedNote_t> tracked_bbox;
	vector< vector<Point> > tracked_points;

	// for twitter
	vector< pair<int, pair<Point, notetype> > > drawing_reserve;

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

			vector<TrackedNote_t> tracked_note = testPattern(frame, frame_broadcast, area_capture_a, template_single);

			if(!tracked_note.empty()){
				for(auto ite = tracked_note.begin(); ite != tracked_note.end(); ++ite){
					Ptr<Tracker> tmp = Tracker::create("KCF");
					tmp->init(frame, (*ite).second);
					tracker.push_back(tmp);
					tracked_bbox.push_back(TrackedNote_t(ite->first, Rect2d(0, 0, ite->second.width, ite->second.height)));

					vector<Point> tmp_vector_point;
					Point tmp_point;
					tmp_point.x = ite->second.y;
					tmp_point.y = ite->second.x;
					tmp_vector_point.push_back(tmp_point);
					tracked_points.push_back(tmp_vector_point);
				}
			}

			vector<TrackedNote_t> tmp_tracked_bbox;
			auto ite_sub = tracked_bbox.begin();
			auto ite_log = tracked_points.begin();
			for(auto ite = tracker.begin(); ite != tracker.end(); ++ite){
				Rect2d tmp_bbox = ite_sub->second;
				(*ite)->update(frame, tmp_bbox);
				if(tmp_bbox.x == 0 && tmp_bbox.y == 0){
				} else if(ite_log->size() == estimate_time){
					// add notes to score(queue)
					int estimated_x_pos = calcLSM(*ite_log);
					cout << "** x= " << estimated_x_pos << endl;
					// rectangle(frame, Rect2d(estimated_x_pos-40, tappoint_y-40, 80, 80), getNotesColor(ite_sub->first), -1);
					estimated_x_pos = adjustEstimatedPos(estimated_x_pos);
					drawing_reserve.push_back(pair<int, pair<Point,notetype> >(num_frame+45, pair<Point,notetype>(Point(estimated_x_pos, tappoint_y), ite_sub->first)));

					tracker.erase(ite--);
					tracked_bbox.erase(ite_sub--);
					tracked_points.erase(ite_log--);
				} else {
					tmp_tracked_bbox.push_back(TrackedNote_t(ite_sub->first, tmp_bbox));
					Point tmp_point;
					tmp_point.x = tmp_bbox.y;
					tmp_point.y = tmp_bbox.x;
					ite_log->push_back(tmp_point);
				}
				++ite_sub;
				++ite_log;
			}
			tracked_bbox.clear();
			copy(tmp_tracked_bbox.begin(), tmp_tracked_bbox.end(), back_inserter(tracked_bbox));
			tmp_tracked_bbox.clear();
			
			// cout << "\033[2J\033[0;0H* tracked bbox" << endl;
			for(auto i : tracked_bbox){
				rectangle(frame, i.second, getNotesColor(i.first), 3);
				// cout << "x:" << (int)(i.second.x) << ", y" << (int)(i.second.y) << ", w" << (int)(i.second.width) << ", h" << (int)(i.second.height) << endl;
			}

			// 補助線の描画
			line(frame, Point(0, 200), Point(1279, 200), Scalar(255, 255, 255), 1);
			line(frame, Point(0, 220), Point(1279, 220), Scalar(255, 255, 255), 1);
			line(frame, Point(0, tappoint_y), Point(1279, tappoint_y), Scalar(255, 255, 255), 3);
			
			// show
			imshow("hoge", frame);
		} else {
			putText(frame, to_string(num_frame), Point(10,40), FONT_HERSHEY_TRIPLEX, 1.5, Scalar(100,100,250), 2, CV_AA);
			imshow("hoge", frame);
		}

		// drawing reserved tapping
		for(auto ite = drawing_reserve.begin(); ite != drawing_reserve.end(); ++ite){
			if(ite->first >= (int)(num_frame) && ite->first <= (int)(num_frame+5)){
				Rect2d tmp;
				tmp.x = ite->second.first.x-40;
				tmp.y = ite->second.first.y-40;
				tmp.width = 80;
				tmp.height = 80;
				if(tmp.x != -40) rectangle(frame_broadcast, tmp, getNotesColor(ite->second.second), -1);
				if(ite->first == (int)(num_frame)) drawing_reserve.erase(ite--);
			}
		}

		// save to movie
		// if(is_store) dst_movie << frame;
		if(is_store) dst_movie << frame_broadcast;
		
		++ num_frame;
	}

	// save to movie
	if(is_store) dst_movie.release();
	
	return 0;
}
