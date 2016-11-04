/*  CS585_Lab2.cpp
    Homework 4
*/

//opencv libraries
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//C++ standard libraries
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <unistd.h>

#define START_MIN 0
#define START_SEC 0
#define REFERENCE_MIN 17
#define REFERENCE_SEC 15
#define NUM_SKELETON_PTS 3
#define FRAME_PROC_RATE 5
#define DISPLAY_VIDEO 1
#define DISPLAY_TMP_VIDEO 0

#define VIDEO_NAME "WithD3.MP4"
#define DATA_FILE_NAME "with_d3_data.csv"

using namespace cv;
using namespace std;

RNG rng(12345);

// Eel object holds where the eel is and it's average curvature
struct eel {
  Point center;
  float curvature;
};
struct dist {
  float dist;
  int index;
};
struct dist_vec {
  vector<dist> dist;
  int index;
};
void process_frame(Mat &frame, bool &first,
                   Point &upper_left, Point &lower_right,
                   Mat &mask, vector<eel> &previous_eels,
                   const char *time_str = "");
void range_threshold(Mat &frame, Mat &dst,
                     uchar l_thresh, uchar u_thresh,
                     Point upper_left = Point(0,0), Point lower_right = Point(0,0),
                     uchar t_val = 255, uchar f_val = 0);
void getBoundingBox(Mat &src, Point &upper_left, Point &lower_right);
void subtract(Mat &src, Mat &mask,
              Point upper_left, Point lower_right);
vector<vector<Point> > getObjects(Mat& src,
                                  int l_thresh = 0, int u_thresh = INT_MAX,
                                  char n_type = 4, bool bounding_box = false, bool center = false);
vector<vector<Point> > getBoundary(Mat& src, vector<vector<Point> > &objects, char n_type = 4);
vector<Point>          getBoundary(Mat& src, vector<Point> &object, char n_type = 4);
vector<vector<Point> > getSkeleton(Mat &src, vector<vector<Point> > &objects);
vector<Point>          getSkeleton(Mat &src, vector<Point> &object);
vector<Point>          getPointsOnSkeleton(vector<Point> skeleton, char sort_by = 0, int num_pts = 5);
vector<float>          getCurvature(vector<Point> skeleton_pts);
float getAvgCurvature(vector<float> curvatures);
int getMinDistanceToBackground(Mat &src, int x, int y, bool check_if_background_first = true);
void setMinMax(int x, int y,
               int &x_min, int &x_max,
               int &y_min, int &y_max);
void setSums(int x, int y,
             int &x_sum, int &y_sum);
bool sortPtsVertical(Point a, Point b);
bool sortPtsHorizontal(Point a, Point b);
bool sortDist(dist a, dist b);
bool sortVecDist(dist_vec a, dist_vec b);
int modulo(int a, int b);
int main()
{
    // --------- SET UP THE VIDEO CAPTURE -------------------------------
    VideoCapture cap;
    cap.open(VIDEO_NAME);

    int num_frames = (int)(cap.get(CV_CAP_PROP_FRAME_COUNT));

    // if not successful, exit program
    if (!cap.isOpened())
    {
        cout << "Cannot open the video file " << VIDEO_NAME << endl;
        return -1;
    }
    // ------------------------------------------------------------------

    // --------- SET UP THE DATA FILE -----------------------------------
    ofstream f (DATA_FILE_NAME);
    if (f.is_open())
    {
      f << "time,change_in_curvature1,change_in_curvature2,..." << endl;
      f.close();
    }
    else {
      cout << "Unable to open file " << DATA_FILE_NAME << endl;
      return 0;
    }
    // ------------------------------------------------------------------

    // --------- SET UP THE DISPLAY WINDOWS -----------------------------
    if (DISPLAY_VIDEO) {
      namedWindow("video", CV_WINDOW_AUTOSIZE);
      namedWindow("tmp", CV_WINDOW_AUTOSIZE);
    }
    // ------------------------------------------------------------------

    // --------- SET UP THE OVER-TIME VARIABLES -------------------------
    Mat frame, mask;
    Point upper_left, lower_right;
    vector<eel> eels;
    bool first = true;
    // ------------------------------------------------------------------

    // --------- GET THE REFERENCE FRAME ---------------------------
    cap.set(CV_CAP_PROP_POS_MSEC, ((REFERENCE_MIN*60) + REFERENCE_SEC)*1000);
    cap >> frame;
    if (frame.empty()) {
      cout << "Bad reference time / frame" << endl;
      return 0;
    }
    process_frame(frame, first, upper_left, lower_right, mask, eels);

    // --------- MAIN LOOP ----------------------------------------------
    // SET UP THE START-TIME OFFSET
    cap.set(CV_CAP_PROP_POS_MSEC, ((START_MIN*60) + START_SEC)*1000);
    int cur_sec = 0;
    while (1)
    {
      // --- GET THE FRAME ------------------------------
      cap >> frame;
      int frame_num = (int)cap.get(CV_CAP_PROP_POS_FRAMES);

      if (frame_num == num_frames) {
        break;
      }
      if (frame.empty()) continue;
      if ((int)cap.get(CV_CAP_PROP_POS_FRAMES) % FRAME_PROC_RATE != 0) continue;
      // ------------------------------------------------

      // --- GET THE TIME OF THE VIDEO STREAM -----------
      int msec = ((int)(cap.get(CV_CAP_PROP_POS_MSEC)));
      int sec, min;
      char time_str[10];
      sec = msec / 1000;
      bool print_time = false;

      // cout << sec << "  ";
      if (sec > cur_sec) {
        print_time = true;
        cur_sec = sec;
      }

      msec = msec - (sec*1000);
      min = sec / 60;
      sec = sec % 60;
      sprintf(time_str, "%i:%i:%i", min, sec, msec);
      if (print_time) cout << time_str << endl;
      // ------------------------------------------------

      process_frame(frame, first, upper_left, lower_right, mask, eels, time_str);

      if (DISPLAY_VIDEO) {
        resize(frame, frame, Size(1000,700));
        imshow("video", frame);
        if (waitKey(30) == 'q') break;
      }
    }
    // ------------------------------------------------------------------

    cap.release();
    return 0;
}

void process_frame(Mat &frame, bool &first,
                   Point &upper_left, Point &lower_right,
                   Mat &mask, vector<eel> &previous_eels,
                   const char *time_str) {
  // DISPLAY THE CURRENT TIME ON THE FRAME
  putText(frame, time_str, Point(frame.cols - 200, frame.rows - 15), FONT_HERSHEY_DUPLEX, 1, Scalar(0,0,200));

  // ------- THIS SECTION CREATES THE BINARY IMAGE OF THE EELS --------
  Mat frame_bw;

  cvtColor(frame, frame_bw, CV_BGR2GRAY);

  // IF THIS IS THE INITIAL CALL, GET THE BOUNDING BOX OF THE TANK
  //    -THE SAME BOUNDING BOX WILL BE USED ON ALL OTHER FRAMES.
  //    -WE'RE ASSUMING THE CAMERA WON'T MOVE MUCH DURING FILMING.
  if (first) {
    //
    Mat tmp;
    range_threshold(frame_bw, tmp, 145, 210, Point(frame.cols / 8,0));
    getBoundingBox(tmp, upper_left, lower_right);
  }
  rectangle(frame, upper_left, lower_right, Scalar(0,0,255), 2,8,0);
  // -----------------------------------

  adaptiveThreshold(frame_bw, frame_bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 5);
  GaussianBlur(frame_bw, frame_bw, Size(17, 17), 0, 0);
  range_threshold(frame_bw, frame_bw, 225, 255, upper_left, lower_right, 0, 255);
  // ------------------------------------------------------------------

  // we're making the assumption that the first frame doesn't
  // have the eel in the frame.
  if (first) mask = frame_bw;
  else {
    // subtract the mask (frame with no eels) from the current frame
    subtract(frame_bw, mask, upper_left, lower_right);

    // `distances[i][j]` would be the distance from the ith eels in the current frame
    // to the jth eel in the last frame (although this is abstracted in the `dist_vec`
    // and `dist` objects)
    vector<dist_vec> distances;
    vector<eel> eels;
    vector<vector<Point> > objects    = getObjects (frame_bw, 400, INT_MAX, 4, true, true);
    vector<vector<Point> > boundaries = getBoundary(frame_bw, objects);
    vector<vector<Point> > skeletons  = getSkeleton(frame_bw, objects);
    frame_bw = Mat::zeros(frame_bw.rows, frame_bw.cols, CV_8UC3);
    for(int i = 0; i < objects.size(); i++) {
      vector<Point> object = objects[i];
      vector<Point> boundary = boundaries[i];
      vector<Point> skeleton = skeletons[i];
      Vec3b color = Vec3b( 0, 200, 0 );
      Vec3b red   = Vec3b( 0, 0, 255 );
      int obj_size = object.size();
      for(int j = 0; j < obj_size-3; j++) {
        int x = object[j].x;
        int y = object[j].y;
        frame_bw.at<Vec3b>(x,y) = color;
      }
      Point obj_up_left = object[obj_size-3];
      Point obj_lw_rght = object[obj_size-2];
      Point center      = object[obj_size-1];

      // this is done so that we can sort the points in the skeleton
      // based on the aspect ratio of the object
      float ratio = ((float)(obj_lw_rght.y - obj_up_left.y)) / ((float)(obj_lw_rght.x - obj_up_left.x));
      char sort_by = (ratio > 1.0) ? 0 : 1;
      vector<Point> skeleton_pts = getPointsOnSkeleton(skeleton, sort_by, NUM_SKELETON_PTS);
      vector<float> curvatures = getCurvature(skeleton_pts);
      float avg_curvature = getAvgCurvature(curvatures);

      eel e;
      e.center = center;
      e.curvature = avg_curvature;
      eels.push_back(e);

      dist_vec d_vec;
      vector<dist> d;
      for(int j = 0; j < previous_eels.size(); j++) {
        float d_tmp = sqrt((float)(pow(center.x - previous_eels[j].center.x,2) + pow(center.y - previous_eels[j].center.y,2)));
        dist dist_struct;
        dist_struct.dist  = d_tmp;
        dist_struct.index = j;
        d.push_back(dist_struct);
      }
      sort(d.begin(), d.end(), sortDist);
      d_vec.dist  = d;
      d_vec.index = i;
      distances.push_back(d_vec);

      for(int j = 0; j < skeleton_pts.size(); j++) {
        circle(frame, Point(skeleton_pts[j].y,skeleton_pts[j].x), 2, Scalar(200, 0, 0), 2, 8, 0);
        circle(frame_bw, Point(skeleton_pts[j].y,skeleton_pts[j].x), 2, Scalar(200, 0, 0), 2, 8, 0);

        // if (j == 0) {
        //   char str_buff[128];
        //   sprintf(str_buff, "%f", avg_curvature);
        //   putText(frame, str_buff, Point(skeleton_pts[j].y,skeleton_pts[j].x), FONT_HERSHEY_DUPLEX, 0.7, Scalar(0,0,255));
        // }
      }
      for(int j = 0; j < boundary.size(); j++) {
        int x = boundary[j].x;
        int y = boundary[j].y;
        frame.at<Vec3b>(x,y) = red;
      }
      for(int j = 0; j < skeleton.size(); j++) {
        int x = skeleton[j].x;
        int y = skeleton[j].y;
        frame.at<Vec3b>(x,y) = red;
        frame_bw.at<Vec3b>(x,y) = red;
      }
    }

    if (DISPLAY_TMP_VIDEO) {
      resize(frame_bw, frame_bw, Size(1000,700));
      imshow("tmp", frame_bw);
    }

    // match the eels in the previous frame to the eels in the current frame
    bool fnd = false;
    ostringstream os;
    for(int i = 0; i < distances.size(); i++) {
      if (distances[0].dist.size() == 0) break;

      sort(distances.begin(), distances.end(), sortVecDist);
      eel cur_eel  = eels[distances[0].index];
      eel last_eel = previous_eels[distances[0].dist[0].index];

      // show the current eels that's matched
      // circle(frame, cur_eel.center, 5, Scalar(255, 255, 0), 2, 8, 0);

      // delete the smallest
      for(int j = 0; j < distances.size(); j++) {
        distances[j].dist.erase(distances[j].dist.begin());
      }

      char tmp_str[16];
      if (!fnd) {
        os << time_str;
        fnd = true;
      }
      sprintf(tmp_str, "%f", (abs(cur_eel.curvature - last_eel.curvature)) / FRAME_PROC_RATE);
      os << ',' << tmp_str;
    }

    // print the results to file
    if (fnd) {
      ofstream f (DATA_FILE_NAME, ios::app);
      if (f.is_open())
      {
        f << os.str() << endl;
        f.close();
      }
      else cout << "Unable to open file " << DATA_FILE_NAME << endl;
    }

    // set the previous eels to the current set
    previous_eels = eels;
  }

  first = false;
}

void subtract(Mat &src, Mat &mask,
              Point upper_left, Point lower_right) {
  if (lower_right.x == 0 && lower_right.y == 0) {
    lower_right.x = src.cols;
    lower_right.y = src.rows;
  }
  for(int i = upper_left.y; i < lower_right.y; i++) {
    for(int j = upper_left.x; j < lower_right.x; j++) {
      uchar val = (src.at<uchar>(i,j) == 255 && mask.at<uchar>(i,j) == 255) ? 0 : src.at<uchar>(i,j);
      src.at<uchar>(i,j) = val;
    }
  }
}

// Threshold's a grayscale image based on a gray-value range
// and within a certain subsection of the image
//
// Parameters:
//  src:         Source image (grayscale)
//  dst:         Destination image
//  l_thresh:    Lower threshold - if pixel is greater than or equal to l_thresh
//                                 less than or equal to u_thresh, set pixel to
//                                 t_val, else set pixel to f_val
//  u_thresh:    Upper threshold - see l_thresh
//  upper_left:  Upper left point of box in image to consider. Pixels outside
//               this box will be set to 0.
//               Default: (0,0)
//  lower_right: Lower right point of box in image to consider. See upper_left.
//               Default: (<lower right corner of src>)
//  t_val:       True value - see l_thresh
//               Default: 255
//  f_val:       False value - see l_thresh
//               Default: 0
void range_threshold(Mat &src, Mat &dst,
                     uchar l_thresh, uchar u_thresh,
                     Point upper_left, Point lower_right,
                     uchar t_val, uchar f_val) {
  if (dst.empty()) dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
  if (lower_right.x == 0 && lower_right.y == 0) {
    lower_right.x = src.cols;
    lower_right.y = src.rows;
  }
  for(int i = 0; i < src.rows; i++) {
    for(int j = 0; j < src.cols; j++) {
      if (i >= upper_left.y && i < lower_right.y &&
          j >= upper_left.x && j < lower_right.x) {
        int p = src.at<uchar>(i,j);
        dst.at<uchar>(i,j) = (p >= l_thresh && p <= u_thresh) ? t_val : f_val;
      }
      else {
        dst.at<uchar>(i,j) = 0;
      }
    }
  }
}

// Get the bounding box of the tank
void getBoundingBox(Mat &src, Point &upper_left, Point &lower_right) {
  vector<int> v(src.rows, 0);
  vector<int> h(src.cols, 0);
  for(int i = 0; i < src.rows; i++) {
    for(int j = 0; j < src.cols; j++) {
      if (src.at<uchar>(i,j) == 255) {
        v[i]++;
        h[j]++;
      }
    }
  }

  int v_thresh = 200;
  int h_thresh = 200;
  int top    = -1;
  int bottom = -1;
  int left   = -1;
  int right  = -1;
  for(int i = 0; i < src.rows; i++) {
    if (top    < 0 && v[i]              > v_thresh) top    = i;
    if (bottom < 0 && v[src.rows-i-1] > v_thresh) bottom = src.rows-i-1;
    if (top >= 0 && bottom >= 0) break;
  }
  for(int i = 0; i < src.cols; i++) {
    if (left  < 0 && h[i]              > h_thresh) left = i;
    if (right < 0 && h[src.cols-i-1] > h_thresh) right = src.cols-i-1;
    if (left >= 0 && right >= 0) break;
  }
  upper_left.x = left;
  upper_left.y = top;
  lower_right.x = right;
  lower_right.y = bottom;
}

// Collects the objects in the binary src image
//
// Params:
//  src:            Source binary image
//  l_thresh:       Objects with less than this number of pixels will be omitted.
//                  Default: 0
//  u_thresh:       Objects with more than this number of pixels will be omitted
//                  Default: INT_MAX
//  n_type:         Boundary method - must be 4 or 8
//                  Default: 4
//  bounding_box:   If true, will calculate the upper left and lower right points of the
//                  bounding box around the object, and store the two points at the end of the
//                  object vector
//                  Default: false
//  center:         If true, will calculate the center of mass of the object and store it at the
//                  end of the object vector
//                  Default: false
vector<vector<Point> > getObjects(Mat& src,
                                  int l_thresh, int u_thresh,
                                  char n_type, bool bounding_box, bool center) {
  vector<vector<Point> > objects;
  vector<int> map (src.rows * src.cols, 0);

  int cur_obj = 1;
  for (int i = 0; i < src.rows; i++){
    for (int j = 0; j < src.cols; j++){
      if (src.at<uchar>(i,j) == 255 && map[i*src.rows + j] == 0) {
        map[i*src.rows + j] = cur_obj;

        stack<Point> s;
        vector<Point> points;
        points.push_back(Point(i,j));

        // also keep track of bounding box for the object
        int x_min = i, x_max = i;
        int y_min = j, y_max = j;
        int x_sum = 0, y_sum = 0;

        s.push(Point(i,j));
        int pts = 0;
        while(!s.empty()) {
          Point p = s.top();
          s.pop();
          int x = p.x;
          int y = p.y;
          // top
          if (x > 0 &&
              src.at<uchar>(x-1,y) == 255 &&
              map[(x-1)*src.rows + y] == 0){
            s.push(Point(x-1, y));
            map[(x-1)*src.rows + y] = cur_obj;
            points.push_back(Point(x-1,y));
            if (bounding_box) setMinMax(x-1, y, x_min, x_max, y_min, y_max);
            if (center) setSums(x-1, y, x_sum, y_sum);
          }
          // left
          if (y > 0 &&
              src.at<uchar>(x,y-1) == 255 &&
              map[x*src.rows + y - 1] == 0){
            s.push(Point(x, y-1));
            map[x*src.rows + y - 1] = cur_obj;
            points.push_back(Point(x,y-1));
            if (bounding_box) setMinMax(x, y-1, x_min, x_max, y_min, y_max);
            if (center) setSums(x, y-1, x_sum, y_sum);
          }
          // right
          if (y < src.cols-1 &&
              src.at<uchar>(x,y+1) == 255 &&
              map[x*src.rows + y + 1] == 0){
            s.push(Point(x, y+1));
            map[x*src.rows + y + 1] = cur_obj;
            points.push_back(Point(x,y+1));
            if (bounding_box) setMinMax(x, y+1, x_min, x_max, y_min, y_max);
            if (center) setSums(x, y+1, x_sum, y_sum);
          }
          // bottom
          if (x < src.rows-1 &&
              src.at<uchar>(x+1,y) == 255 &&
              map[(x+1)*src.rows + y] == 0){
            s.push(Point(x+1, y));
            map[(x+1)*src.rows + y] = cur_obj;
            points.push_back(Point(x+1,y));
            if (bounding_box) setMinMax(x+1, y, x_min, x_max, y_min, y_max);
            if (center) setSums(x+1, y, x_sum, y_sum);
          }
          // top left
          if (n_type == 8) {
            if (x > 0 &&
                y > 0 &&
                src.at<uchar>(x-1,y-1) == 255 &&
                map[(x-1)*src.rows + (y-1)] == 0){
              s.push(Point(x-1, y-1));
              map[(x-1)*src.rows + (y-1)] = cur_obj;
              points.push_back(Point(x-1,y-1));
              if (bounding_box) setMinMax(x-1, y-1, x_min, x_max, y_min, y_max);
              if (center) setSums(x-1, y-1, x_sum, y_sum);
            }
            // top right
            if (x > 0 &&
                y < src.cols-1 &&
                src.at<uchar>(x-1,y+1) == 255 &&
                map[(x-1)*src.rows + (y+1)] == 0){
              s.push(Point(x-1, y+1));
              map[(x-1)*src.rows + (y+1)] = cur_obj;
              points.push_back(Point(x-1,y+1));
              if (bounding_box) setMinMax(x-1, y+1, x_min, x_max, y_min, y_max);
              if (center) setSums(x-1, y+1, x_sum, y_sum);
            }
            // bottom left
            if (x < src.rows-1 &&
                y > 0 &&
                src.at<uchar>(x+1,y-1) == 255 &&
                map[(x+1)*src.rows + (y-1)] == 0){
              s.push(Point(x+1, y-1));
              map[(x+1)*src.rows + (y-1)] = cur_obj;
              points.push_back(Point(x+1,y-1));
              if (bounding_box) setMinMax(x+1, y-1, x_min, x_max, y_min, y_max);
              if (center) setSums(x+1, y-1, x_sum, y_sum);
            }
            // bottom right
            if (x < src.rows-1 &&
                y < src.cols-1 &&
                src.at<uchar>(x+1,y+1) == 255 &&
                map[(x+1)*src.rows + (y+1)] == 0){
              s.push(Point(x+1, y+1));
              map[(x+1)*src.rows + (y+1)] = cur_obj;
              points.push_back(Point(x+1,y+1));
              if (bounding_box) setMinMax(x+1, y+1, x_min, x_max, y_min, y_max);
              if (center) setSums(x+1, y+1, x_sum, y_sum);
            }
          }
        }
        if (points.size() < l_thresh ||
            points.size() > u_thresh) {
          for(int k = 0; k < points.size(); k++) {
            int x = points[k].x;
            int y = points[k].y;
            map[x*src.rows + y] = -1;
          }
        }
        else {
          cur_obj++;
          if (bounding_box) {
            Point upper_left(y_min, x_min);
            Point lower_right(y_max, x_max);
            points.push_back(upper_left);
            points.push_back(lower_right);
          }
          if (center) {
            int s = points.size();
            int x_bar = x_sum / s;
            int y_bar = y_sum / s;
            Point c(x_bar, y_bar);
            points.push_back(c);
          }
          objects.push_back(points);
        }
      }
    }
  }
  return objects;
}

// Helper function for getObjects to keep track of bounding box of object
//
// Params:
//  x:     Current x value
//  y:     Current y value
//  x_min: Current minimum x value
//  x_max: Current maximum x value
//  y_min: Current minimum y value
//  y_max: Current maximum y value
void setMinMax(int x, int y,
               int &x_min, int &x_max,
               int &y_min, int &y_max) {
  if (x < x_min) x_min = x;
  if (x > x_max) x_max = x;
  if (y < y_min) y_min = y;
  if (y > y_max) y_max = y;
}

// Helper function for getObjects to calculate centroid of object
//
// Params:
//  x:     Current x value
//  y:     Current y value
//  x_sum: Current sum of y values (used to calculate x_bar)
//  y_sum: Current sum of x values (used to calculate y_bar)
void setSums(int x, int y,
             int &x_sum, int &y_sum) {
  x_sum += y;
  y_sum += x;
}

// Calculates the boundaries of objects in the binary src image
//
// Params:
//  src:     Source binary image
//  objects: The set of objects found from the getObjects function
//  n_type:  Neighborhood type to use (4 or 8)
//           Default: 4
vector<vector<Point> > getBoundary(Mat &src, vector<vector<Point> > &objects, char n_type) {
  vector<vector<Point> > boundaries;
  int num_object = objects.size();
  for(int i = 0; i < num_object; i++) {
    vector<Point> object = objects[i];
    vector<Point> boundary = getBoundary(src, object, n_type);
    boundaries.push_back(boundary);
  }
  return boundaries;
}

// Calculates the boundary of an object in the binary src image
//
// Params:
//  src:     Source binary image
//  object:  An objects found from the getObjects function
//  n_type:  Neighborhood type to use (4 or 8)
//           Default: 4
vector<Point> getBoundary(Mat& src, vector<Point> &object, char n_type) {
    // vector<Point> points = objects[i];
  vector<Point> boundary;
  Point origin = object[0];
  Point cur_point = origin;
  bool first = true;  // just need a way to get into the loop
  int fnd;
  int x, y;
  while (cur_point != origin || first) {
    if (first) {
      fnd = 1;
      first = false;
    }
    x = cur_point.x;
    y = cur_point.y;
    int x_trgt, y_trgt;
    fnd = modulo(fnd-1, n_type);
    bool found = false;
    if (n_type == 4) {
      for (int j = 0; j < n_type; j++) {
        int cur_check = (fnd+j) % n_type;
        switch (cur_check) {
          case 0:
            if (y > 0) {
              if (src.at<uchar>(x, y-1) == 255) {
                cur_point = Point(x, y-1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 1:
            if (x > 0) {
              if (src.at<uchar>(x-1, y) == 255) {
                cur_point = Point(x-1, y);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 2:
            if (y < src.cols-1) {
              if (src.at<uchar>(x, y+1) == 255) {
                cur_point = Point(x, y+1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 3:
            if (x < src.rows-1) {
              if (src.at<uchar>(x+1, y) == 255) {
                cur_point = Point(x+1, y);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
        }
        if (found) break;
      }
    }
    else if (n_type == 8) {
      for (int j = 0; j < n_type; j++) {
        int cur_check = (fnd+j) % n_type;
        switch (cur_check) {
          case 0:
            if (y > 0) {
              if (src.at<uchar>(x, y-1) == 255) {
                cur_point = Point(x, y-1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 1:
            if (y > 0 && x > 0) {
              if (src.at<uchar>(x-1, y-1) == 255) {
                cur_point = Point(x-1, y-1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 2:
            if (x > 0) {
              if (src.at<uchar>(x-1, y) == 255) {
                cur_point = Point(x-1, y);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 3:
            if (x > 0 && y < src.cols-1) {
              if (src.at<uchar>(x-1, y+1) == 255) {
                cur_point = Point(x-1, y+1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 4:
            if (y < src.cols-1) {
              if (src.at<uchar>(x, y+1) == 255) {
                cur_point = Point(x, y+1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 5:
            if (y < src.cols-1 && x < src.rows-1) {
              if (src.at<uchar>(x+1, y+1) == 255) {
                cur_point = Point(x+1, y+1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 6:
            if (x < src.rows-1) {
              if (src.at<uchar>(x+1, y) == 255) {
                cur_point = Point(x+1, y);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
          case 7:
            if (x < src.rows-1 && y > 0) {
              if (src.at<uchar>(x+1, y-1) == 255) {
                cur_point = Point(x+1, y-1);
                boundary.push_back(cur_point);
                fnd = cur_check;
                found = true;
              }
            }
            break;
        }
        if (found) break;
      }
    }
  }
  return boundary;
}

// Calculates the skeletons of objects in the binary src image
//
// Params:
//  src:     Source binary image
//  objects: The set of objects found from the getObjects function
vector<vector<Point> > getSkeleton(Mat &src, vector<vector<Point> > &objects) {
  vector<vector<Point> > skeletons;
  int num_object = objects.size();
  for(int i = 0; i < num_object; i++) {
    vector<Point> object = objects[i];
    vector<Point> skeleton = getSkeleton(src, object);
    skeletons.push_back(skeleton);
  }
  return skeletons;
}

// Calculates the skeleton of an object in the binary src image
//
// Params:
//  src:     Source binary image
//  objects: Object in the binary source image
vector<Point> getSkeleton(Mat &src, vector<Point> &object) {
  vector<Point> skeleton;

  // using '-3' here because the last three points of the object are two the corners of
  // the skeleton's bounding box and the center of the object
  int object_len = object.size()-3;
  for(int i = 0; i < object_len; i++) {
    int x = object[i].x;
    int y = object[i].y;

    int dist = getMinDistanceToBackground(src, x, y, false);
    if (dist < getMinDistanceToBackground(src, x-1, y  ) ||
        dist < getMinDistanceToBackground(src, x  , y-1) ||
        dist < getMinDistanceToBackground(src, x+1, y  ) ||
        dist < getMinDistanceToBackground(src, x  , y+1)) continue;
    skeleton.push_back(object[i]);
  }

  return skeleton;
}

// Helper function for getSkeleton. Calculates the minimum distance from
// a pixel to the background of the object.
//
// Params:
//  src:                       Source binary image
//  x:                         x-coordinate of pixel
//  y:                         y-coordinate of pixel
//  check_if_background_first: If true, will first check if the pixel is
//                             itself a background pixel.
//                             Default: true
int getMinDistanceToBackground(Mat &src, int x, int y, bool check_if_background_first) {
  if (check_if_background_first && (x < 0 || y < 0 ||
                                    x == src.cols || y == src.rows ||
                                    src.at<uchar>(x,y) == 0)) return 0;

  int x_tmp, y_tmp;
  int dist[] = {1,1,1,1};

  // get distance to east
  x_tmp = x; y_tmp = y;
  while (--x_tmp >= 0 && src.at<uchar>(x_tmp,y_tmp) != 0) dist[0]++;

  // get distance to north
  x_tmp = x; y_tmp = y;
  while (--y_tmp >= 0 && src.at<uchar>(x_tmp,y_tmp) != 0) dist[1]++;

  // get distance to west
  x_tmp = x; y_tmp = y;
  while (++x_tmp < src.cols && src.at<uchar>(x_tmp,y_tmp) != 0) dist[2]++;

  // get distance to south
  x_tmp = x; y_tmp = y;
  while (++y_tmp < src.rows && src.at<uchar>(x_tmp,y_tmp) != 0) dist[3]++;

  int m1 = min(dist[0], dist[1]);
  int m2 = min(dist[2], dist[3]);
  int min_dist = min(m1,m2);
  return min_dist;
}

// Calculates some number of evenly distributed points along the
// skeleton of the image.
//
// Params:
//  skeleton: Set of points making up the skeleton of an object
//  sort_by:  x-coordinate of pixel
//            Default: 0
//  num_pts:  Number of points along skeleton to find
//            Default: 5
vector<Point> getPointsOnSkeleton(vector<Point> skeleton, char sort_by, int num_pts) {
  int size = skeleton.size();
  vector<Point> points;

  // sort the skeleton points either vertically or horizontally
  if (sort_by == 1) sort(skeleton.begin(), skeleton.end(), sortPtsVertical);
  else if (sort_by == 0) sort(skeleton.begin(), skeleton.end(), sortPtsHorizontal);

  // set the number of points to grab to 2 less, because the first
  // and last will be the endpoints of the skeleton
  if (num_pts > size) num_pts = size;
  num_pts -= 2;

  // add the first point
  points.push_back(skeleton[0]);

  // determine the number of points in between each representative point
  int step = size / (num_pts + 1);
  for(int i = 0, cur = step; i < num_pts; i++, cur += step) {
    points.push_back(skeleton[cur]);
  }

  // add the last point
  points.push_back(skeleton[size-1]);

  return points;
}

// Sort a vector of points vertically (based on y-value first, then x-value)
bool sortPtsVertical(Point a, Point b) {
  bool ret = false;
  if (a.y < b.y || (a.y == b.y && a.x < b.x)) ret = true;
  return ret;
}

// Sort a vector of points horizontally (based on x-value first, then y-value)
bool sortPtsHorizontal(Point a, Point b) {
  bool ret = false;
  if (a.x < b.x || (a.x == b.x && a.y < b.y)) ret = true;
  return ret;
}

// Calculate the curvatures of a set of points. There will be two
// fewer curvatures than points given, since it takes three points
// to calculate a curvature value. The first curvature value is
// the curvature given by the vector between points 0 and 1 and
// the vector between points 1 and 2, The second curvature value
// is the curvature given by the vector between points 1 and 2 and
// the vector between points 2 and 3, and so on.
//
// Params:
//  skeleton_pts: vector of representative points along a skeleton
vector<float> getCurvature(vector<Point> skeleton_pts) {
  int len = skeleton_pts.size() - 1;
  vector<float> curvature;

  for(int i = 1; i < len; i++) {
    float cur_mag = sqrt((float)(pow(skeleton_pts[i].x - skeleton_pts[i-1].x, 2) + pow(skeleton_pts[i].y - skeleton_pts[i-1].y, 2)));
    float nxt_mag = sqrt((float)(pow(skeleton_pts[i+1].x - skeleton_pts[i].x, 2) + pow(skeleton_pts[i+1].y - skeleton_pts[i].y, 2)));
    float cur_x = ((float)(skeleton_pts[i].x - skeleton_pts[i-1].x)) / cur_mag;
    float cur_y = ((float)(skeleton_pts[i].y - skeleton_pts[i-1].y)) / cur_mag;
    float nxt_x = ((float)(skeleton_pts[i+1].x - skeleton_pts[i].x)) / nxt_mag;
    float nxt_y = ((float)(skeleton_pts[i+1].y - skeleton_pts[i].y)) / nxt_mag;
    float rslt_x = cur_x - nxt_x;
    float rslt_y = cur_y - nxt_y;
    float rslt = (float)(pow(rslt_x, 2) + pow(rslt_y, 2));
    curvature.push_back(rslt);
  }

  return curvature;
}

// Averages the values in the given vector
//
// Params:
//  curvatures: vector of values to average
float getAvgCurvature(vector<float> curvatures) {
  int len = curvatures.size();
  float sum = 0.0;
  for(int i = 0; i < len; i++) sum += curvatures[i];

  return sum / len;
}

// Sort the `dist` (distance object) based on the `dist` value
bool sortDist(dist a, dist b) {
  return (a.dist < b.dist) ? true : false;
}

// Sort the `dist_vec` (vector of `dist`-objects object) based on the first value
// in the `dist` array. (I know, I'm calling way too many different things `dist` here... :( )
bool sortVecDist(dist_vec a, dist_vec b) {
  return ((a.dist)[0].dist < (b.dist)[0].dist) ? true : false;
}

// a real module function (i.e. it can properly handle negatives)
int modulo(int a, int b) {
  const int result = a % b;
  return result >= 0 ? result : result + b;
}
