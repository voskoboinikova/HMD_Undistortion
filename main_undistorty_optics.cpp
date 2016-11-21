#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
#include <Eigen/LU>

using namespace Eigen;
using namespace std;

struct mis{
int fig_id;
int xlim_min;
int xlim_max;
int ylim_min;
int ylim_max;
int quiverscale;
int h;
} misc;


struct KR_model{
    VectorXd alpha; //132x1
    float sigma;
    float lambda;
    float gamma;
    MatrixXd center; //4x132
} KR_model0_ytr[4]; //for now 4 is here, but later it must be dynamically changed


struct KR_model_vs{
    VectorXd lambda_list; //1x6
    VectorXd sigma_list; //1x9
    MatrixXd xtr; //4x132
    MatrixXd ytr; //4x132
    VectorXd ytrain_mean;  //4x1
    VectorXd ytrain_std; //4x1
    VectorXd xtrain_mean; //4x1
    VectorXd xtrain_std; //4x1
    bool USE_IWLS; //0
    bool NORMALIZE_DATA; //1
    KR_model KR_model0_ytr[4]; //4x1 cell
    bool DISTORT_UVST; //1
} KR_model_optical;


/*
void openUbitrack3x3MatrixCalib(float** file_K, float** K_raw){
		
	std::ifstream fid("cameraintrinsics.txt");	
	std::string tline;
	if( fid.is_open() ){
		getline(fid,tline);
		std::cout << tline << std::endl;
		fid.close();
	}
	else{
		std::cout << "unable to open file" << std::endl;
	}
	//sscanf( tline, '%*d %*s %*d %*d %*d %*d %*d %*d %f%f%f%f%f%f%f%f%f' );

};	
*/

void openUbitrack3DPositionList(char* filename, int* N, float* A){
	int D = 3;
 
	FILE * fp;
	fp = fopen(filename, "r");
	fscanf(fp, "22 serialization::archive 9 0 0 %*f 0 0 %d 0 0 0", N);
	int Num = *N;
		
	for(int i=0; i<Num; i++){
		int ind = i*D;
		fscanf(fp, "%f %f %f", &(A[ind]), &(A[ind+1]), &(A[ind+2]));
	}
	fclose(fp);
}

void openUbitrack2DPositionList(char* filename, int* N, float* A){
	int D = 2;
 
	FILE * fp;
	fp = fopen(filename, "r");
	fscanf(fp, "22 serialization::archive 9 0 0 %*f 0 0 %d 0 0 0", N);
	int Num = *N;
		
	for(int i=0; i<Num; i++){
		int ind = i*D;
		fscanf(fp, "%f %f", &(A[ind]), &(A[ind+1]));
	}
	fclose(fp);
}

void openUbitrack6DPoseCalib(char* filename, float* A ){
	FILE * fp;
	fp = fopen(filename, "r");
	fscanf(fp, "%*d%*s%*d%*d%*d%f%*d%*d%*d%*d%f%f%f%f%*d%*d%f%f%f", &(A[0]), &(A[1]), &(A[2]), &(A[3]), &(A[4]), &(A[5]), &(A[6]), &(A[7]));
	fclose(fp);

	/*
	std::cout << "printing A4" << std::endl;
	for(int i=0; i<8; i++){
		std::cout << A[i] << std::endl;
	}
	std::cout << " \n" << std::endl;
	*/
}


void ubitrackQuat2Mat(float* q, float** R){
    //std::cout << "ubitrackQuat2Mat" << std::endl;

    Matrix4d I;
	I << 0,0,-1,0,  0, -1, 0, 0,  1, 0, 0, 0,   0, 0, 0, 1;
	//std::cout << "Here is the matrix I:\n" << I << std::endl;

    Vector4d qu;
	qu<< q[0], q[1], q[2], q[3];
	//std::cout << "Here is the vector q:\n " << qu << std::endl;

    Vector4d mult;
	mult = I*qu;
	//std::cout << "Here is the vector mult=I*q:\n" << mult << std::endl;
	//std::cout << " \n " << std::endl; 

	//std::cout << "quat.x() = mult(1);\n quat.y() = mult(2); \n quat.z() = mult(3); \n quat.w() = mult(0);" << std::endl;
	Quaterniond quat;
	 quat.x() = mult(1);
	 quat.y() = mult(2);
	 quat.z() = mult(3);
	 quat.w() = mult(0);
	

	Matrix3d quat2dc;
	quat2dc = (quat.normalized().toRotationMatrix()).transpose();
	//std::cout << "Here is the matrix quat2dc:\n" << quat2dc << std::endl;
	//std::cout << " \n " << std::endl;

	Matrix3d mat;
	mat << -1, 0, 0,  0, -1, 0, 0, 0, 1;

	Matrix3d Result;
	Result = mat*quat2dc.transpose();
	
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			R[i][j] = Result(i,j);
		}
	}

	

}

void loadUbitrackPose0(char* filename, float** R,  float* t){
    //std::cout << "loadUbitrackPose0" << std::endl;
    float* tmp = new float[8];
	openUbitrack6DPoseCalib(filename, tmp);
    float tstamp = tmp[0];
    float* q = new float[4];
	q[0] = tmp[1]; q[1] = tmp[2]; q[2] = tmp[3]; q[3] = tmp[4];
	
	ubitrackQuat2Mat(q, R);

	t[0] = tmp[5]; t[1] = tmp[6]; t[2] = tmp[7];
}


void loadDataSetEigen(char* dir, float K_E[3][3], mis misc, int* Num, Eigen::MatrixXd& X_W_eig, Eigen::MatrixXd& Y_eig, Eigen::MatrixXd&  X_distorted_eig,
                        Eigen::Vector3d& t_E0W_eig, Eigen::Vector3d& t_WE0_eig,
                        Eigen::MatrixXd&  G_E_eig, Eigen::MatrixXd&  G_distorted_E_eig){

    cout << "\n" << "loadDataSetEigen() " << endl;
    /********************************************************************/


    //  filename = strcat(dir,'3dPointsOnGrid.txt');
    // p3D=openUbitrack3DPositionList(filename);

    char filename1[60] = "";
    strcat(filename1, dir);
    strcat(filename1, "3dPointsOnGrid.txt");
    //std::cout << filename1 << std::endl;
    int D = 3;
    int N;
    int Nmax = 100;
    float* A = new float[Nmax*D];
    openUbitrack3DPositionList(filename1, &N, A);
    Num = &N;




    /*******arrays for internal computations******/
    MatrixXd p3D_eig(3,N);
    MatrixXd p2D_E_eig(2,N), p2D_distorted_E_eig(2,N);

    p3D_eig = MatrixXd::Zero(3,N);
    p2D_E_eig = MatrixXd::Zero(2,N);
    p2D_distorted_E_eig = MatrixXd::Zero(2,N);

    Matrix3d R_E0W_eig = Matrix3d::Zero();
    //Vector3d t_E0W_eig, t_WE0_eig;
    Matrix3d R_ME0_eig = Matrix3d::Zero();
    Vector3d t_ME0_eig = Vector3d::Zero();

    Matrix4d P_E0W_eig, P_ME0_eig, P_MW_eig;
    P_E0W_eig = Matrix4d::Zero(4,4);
    P_ME0_eig = Matrix4d::Zero(4,4);
    P_MW_eig = Matrix4d::Zero(4,4);

    Matrix3d R_MW_eig;
    Vector3d t_MW_eig;
    Matrix3d K_E_eig;
    K_E_eig = Matrix3d::Zero();
    R_MW_eig = Matrix3d::Zero();
    t_MW_eig = Vector3d::Zero();

    //MatrixXd X_W_eig(3,N), Y_eig(3,N);
    //MatrixXd G_E_eig(3,N), G_distorted_E_eig(3,N), X_distorted_eig(3,N);
    //X_W_eig = MatrixXd::Zero(3,N);
    //Y_eig = MatrixXd::Zero(3,N);
    //X_distorted_eig = MatrixXd::Zero(3,N);
    //G_E_eig = MatrixXd::Zero(3,N);
    //G_distorted_E_eig = MatrixXd::Zero(3,N);

    MatrixXd aux = MatrixXd::Zero(3,N);
    /*********************************************/



    //p3D[D,N]
    for(int j=0; j<N;j++){
        int ind = j*D;
        p3D_eig(0,j) = A[ind];
        p3D_eig(1,j) = A[ind+1];
        p3D_eig(2,j) = A[ind+2];
    }

    //std::cout << "printing p3D: " << "\n" << p3D_eig.transpose() << "\n" << std::endl;
    /*********************************************************************/






    /*********************************************************************/
    //p2D_E = openUbitrack2DPositionList(filename);
    //p2D_E(2,:) = misc.h-1-p2D_E(2,:);

    char filename2[60] = "";
    strcat(filename2, dir);
    strcat(filename2, "2dPointsOnViewPoint.txt");
    //std::cout << filename2 << std::endl;
    D = 2;
    float* A2 = new float[Nmax*D];
    openUbitrack2DPositionList(filename2, &N, A2);

    //p2D_E[D,N]
    for(int i=0; i<N; i++){
        int ind = i*D;
        p2D_E_eig(0,i) = A2[ind];
        p2D_E_eig(1,i) = misc.h - 1.0 - A2[ind+1];
    }

    //std::cout << "printing p2D_E: " << "\n" << p2D_E_eig.transpose() << "\n" << std::endl;
    /*********************************************************************/

    /********************************************************************/
    // p2D_distorted_E=openUbitrack2DPositionList(filename);
    //p2D_distorted_E(2,:) = misc.h-1-p2D_distorted_E(2,:);

    char filename3[60] = "";
    strcat(filename3, dir);
    strcat(filename3, "2dPointsOnViewPointThroughHMD.txt");
    //std::cout << filename3 << std::endl;
    D=2;
    float* A3 = new float[Nmax*D];
    openUbitrack2DPositionList(filename3, &N, A3);

    for(int j=0; j<N;j++){
        int ind = j*D;
        p2D_distorted_E_eig(0,j) = A3[ind];
        p2D_distorted_E_eig(1,j) = misc.h - 1.0 - A3[ind+1];
    }

   // std::cout <<"N=" << N << std::endl;
   // std::cout << "printing p2D_distorted_E_eig" << "\n" << p2D_distorted_E_eig.transpose() << "\n" <<std::endl;
    /********************************************************************/


    /********************************************************************/
    // [R_E0W, t_E0W]=loadUbitrackPose0(filename);
    //t_WE0 = -R_E0W'*t_E0W;

    char filename4[60] = "";
    strcat(filename4, dir);
    strcat(filename4,"P_HMDCam2IDS.txt");
    //std::cout << filename4 << std::endl;

    float** R_E0W = new float*[3];
    for(int i=0; i<3; i++){
        R_E0W[i] = new float[3];
    }

    float* t_E0W = new float[3];
    float* t_WE0 = new float[3];
    loadUbitrackPose0(filename4, R_E0W, t_E0W);

    //t_WE0 = -R_E0W'*t_E0W;
    //convert the achieved matrices into Eigen format
    for(int i=0; i<3;i++){
        R_E0W_eig(i,0) = R_E0W[i][0];
        R_E0W_eig(i,1) = R_E0W[i][1];
        R_E0W_eig(i,2) = R_E0W[i][2];
    }
    std::cout << "\n" << "Here is the matrix R_E0W_eig: \n " << R_E0W_eig << std::endl;

    t_E0W_eig << t_E0W[0], t_E0W[1], t_E0W[2];
    std::cout << "\n \n" << "Here is the vector t_E0W_eig: \n " << t_E0W_eig << std::endl;

    t_WE0_eig =  -R_E0W_eig.transpose()*t_E0W_eig;
    //std::cout << "Here is the vector t_WE0_eig: \n " << t_WE0_eig << std::endl;

    //transform t_WE0_eig to float array
    t_WE0[0] = t_WE0_eig(0); t_WE0[1] = t_WE0_eig(1); t_WE0[2] = t_WE0_eig(2);

    //std::cout << "\n " << std::endl;
    /**********************************************************************/

    /*********************************************************************/
    //[R_ME0, t_ME0]=loadUbitrackPose0(filename);

    char filename5[60] = "";
    strcat(filename5, dir);
    strcat(filename5, "P_IDS2Marker.txt");
    //std::cout << filename5 << std::endl;

    float** R_ME0 = new float*[3];
    for(int i=0; i<3; i++){
        R_ME0[i] = new float[3];
    }

    float* t_ME0 = new float[3];

    loadUbitrackPose0(filename5, R_ME0, t_ME0);

    for(int i=0; i<3; i++){
        R_ME0_eig(i,0) = R_ME0[i][0];
        R_ME0_eig(i,1) = R_ME0[i][1];
        R_ME0_eig(i,2) = R_ME0[i][2];
        t_ME0_eig(i) = t_ME0[i];
    }
    //std::cout << "Here is the matrix R_ME0_eig: \n " << R_ME0_eig << std::endl;
    //std::cout << "Here is the vector t_ME0_eig: \n " << t_ME0_eig << std::endl;
    //std::cout << "\n" << std::endl;
    /********************************************************************/


    /*********************************************************************/
    //X_W = R_MW*p3D+repmat(t_MW,1,N);
    //Y=repmat(t_E0W,1,N);
    //G_E = R_ME0*p3D+repmat(t_ME0,1,N);
    //G_distorted_E = K_E\[p2D_distorted_E;ones(1,N)];
    //X_distorted=R_E0W*G_distorted_E + repmat(t_E0W,1,N);
    //std::cout << "Computing X_W, Y,G_E, G_distorted_E, X_distorted " << std::endl;

    P_E0W_eig << R_E0W_eig, t_E0W_eig, 0, 0, 0, 1;
    P_ME0_eig << R_ME0_eig, t_ME0_eig, 0, 0, 0, 1;
    P_MW_eig = P_E0W_eig*P_ME0_eig;

    //Eigen: P.topLeftCorner(rows, cols)  Matlab: P(1:rows, 1:cols)
    //Eigen: P.topRightCorner(rows, cols) Matlab: P(1:rows, end-cols+1:end)
    R_MW_eig << P_MW_eig.topLeftCorner(3,3);
    t_MW_eig << P_MW_eig.topRightCorner(3,1);

    for(int i=0; i<3; i++){
        K_E_eig(i,0) = K_E[i][0];
        K_E_eig(i,1) = K_E[i][1];
        K_E_eig(i,2) = K_E[i][2];
    }

    //what to do if I cant write N as a parameter to replicate(instead of 44)?
    X_W_eig = R_MW_eig * p3D_eig + t_MW_eig.replicate<1,44>();
    Y_eig = t_E0W_eig.replicate<1,44>();
    G_E_eig = R_ME0_eig*p3D_eig + t_ME0_eig.replicate<1,44>();

    for(int j=0; j<N; j++){
        aux(0, j) = p2D_distorted_E_eig(0,j);
        aux(1,j) = p2D_distorted_E_eig(1,j);
        aux(2,j) = 1;
    }

    G_distorted_E_eig = K_E_eig.inverse() * aux;
    X_distorted_eig = R_E0W_eig*G_distorted_E_eig + t_E0W_eig.replicate<1,44>();

    //std::cout << "\n" << std::endl;
    /**********************************************************************************/



    //to transform Eigen final values into float**, if needed
    //to free memory for auxilary arrays, matrices
    //to read about sending Eigen as a parameter to a function




    //free memory
    delete[] A;
    delete[] A2;
    delete[] A3;
    delete[] t_E0W;
    delete[] t_WE0;


}


/*
         function [XS_W0,XS_S0] = intersetRayWithPlane(X0,Y0,t_PW,R_SW)
                R_WS = R_SW';
                t_WP = -R_SW'*t_PW; %the eyeball center
                r3=R_WS(3,:)';
                dx=X0-Y0; % 3D grid points in E0
                a=( t_PW'*r3 - Y0'*r3)./(dx'*r3); % a: Nx1
                XS_W0= Y0+dx.*repmat(a',3,1);% 3D points on the virtual display in W
                XS_S0=R_WS*XS_W0+repmat(t_WP,1,size(XS_W0,2));% in S, so XS_S(3,:) are all 0
         end
*/


void intersectRayWithPlane(Eigen::MatrixXd& X0, Eigen::MatrixXd& Y0,  Eigen::Vector3d& t_PW,  Eigen::Matrix3d& R_SW, Eigen::MatrixXd& XS_W0, Eigen::MatrixXd& XS_S0){
    int N = X0.cols();

    Matrix3d R_WS;
    Vector3d t_WP, r3;
    MatrixXd dx = MatrixXd::Zero(3,N);

    VectorXd nom(N), denom(N);
    VectorXd a = VectorXd::Zero(N);

    R_WS = R_SW.transpose();
    t_WP = -R_SW.transpose() * t_PW; //the eyeball center
    r3 = R_WS.row(2);

    dx = X0 - Y0; //3D points in E0

    nom = (t_PW.transpose() * r3).replicate(N,1) - Y0.transpose() * r3;
    denom = dx.transpose() * r3;
    a = nom.array() / denom.array();

    //3D points on the virtual display in W
    XS_W0 = Y0 +  dx.cwiseProduct( a.transpose().replicate(3,1) );
    XS_S0 =  R_WS*XS_W0 + t_WP.replicate(1,N);
}




void standDev(Eigen::MatrixXd& a, Eigen::VectorXd& s){
    int ncols = a.cols();
    int nrows = a.rows();

    for(int i=0; i<nrows; i++){
        VectorXd ai = a.row(i);
        float mu = ai.mean();
        float sum = 0;
        for(int j=0; j<ncols; j++){
            sum += (ai(j) - mu)*(ai(j) - mu);
        }
        s(i) = sqrt(sum / (ncols-1));
    }
}


void normalize_dist(Eigen::MatrixXd& x, Eigen::VectorXd& m, Eigen::VectorXd& std, Eigen::MatrixXd& x_n){
    /*
    function x_n = normalize_dist(x,m,std)
        assert( size(x,1)==size(m,1) && size(m,1)==size(std,1), 'input should be x: DxN, m:Dx1, and s:Dx1' )
        N=size(x,2);
        x_n = (x - repmat(m,1,N))./repmat(std,1,N);
    end
    */

    //dst = src.replicate(n,m);

    int ncols = x.cols();
    int nrows = x.rows();
    int N = ncols;

    //std::cout << "ncols = " << ncols << std::endl;
    //std::cout << "nrows = " << nrows << std::endl;
    x_n = (x - m.replicate(1,N)).array() / std.replicate(1,N).array();
}


void unnormalize_dist(Eigen::MatrixXd& x_n, Eigen::VectorXd& m, Eigen::VectorXd& ssqrt, Eigen::MatrixXd& x){
    ///function x = unnormalize_dist(x_n,m,ssqrt)
    ///     assert( size(x_n,1)==size(m,1) && size(m,1)==size(ssqrt,1), 'input should be x: DxN, m:Dx1, and s:Dx1' )
    ///     N=size(x_n,2);
    ///     x = ( x_n.*repmat(ssqrt,1,N) ) + repmat(m,1,N);
    ///end
    int N = x_n.cols();
    MatrixXd mat1 = x_n.array() * ssqrt.replicate(1,N).array();
    MatrixXd mat2 = m.replicate(1,N);
    cout << "mat1.cols = " << mat1.cols() <<", mat1.rows()=" << mat1.rows() << endl;
    cout << "mat2.cols = " << mat2.cols() << ", mat2.rows()=" << mat2.rows() << endl;
    cout << "x.cols = " << x.cols() << ", x.rows() = " << x.rows() << endl;
    x = mat1 + mat2;
}




void linspace(float low, float high, float size, Eigen::VectorXd& out){
    if(low>high) std::swap(low,high);
    //y = 10 .^ linspace(d1, d2, n);

    Eigen::VectorXd v = VectorXd::Zero(size);
    v.setLinSpaced(size,low,high);        // v = linspace(low,high,size)'
    out = 10 * VectorXd::Ones(size);
    for(int i = 0; i<size; i++){
        out(i) = pow(10, v(i));
    }

}



void calcVecMedian(Eigen::VectorXd& vec, float* med){
    int size = vec.size();
    int sizehalf = size/2;
    std::sort(vec.data(), vec.data()+vec.size());
    if (fmod(size,2) == 0 ){
      *med = (vec(sizehalf - 1) + vec(sizehalf)) / 2;
    }
    else{
        *med = vec(sizehalf);
    }
    //cout << "med = " << *med << endl;

}


void IWLS_train(Eigen::MatrixXd& x, Eigen::VectorXd& y, Eigen::VectorXd& sigma_list, Eigen::VectorXd& lambda_list, Eigen::VectorXd& gamma_list, int b, KR_model* KR_model0_ytr){
    //cout << "x: " << "\n" << x.transpose() << endl;
    //cout << "y: " << "\n" << y.transpose() << endl;


    ///sigma_list:     (OPTIONAL) Gaussian width
    ///                If sigma_list is a vector, one of them is selected by cross validation.
    ///lambda_list:    (OPTIONAL) regularization parameter
    ///                If lambda_list is a vector, one of them is selected by cross validation.
    ///gamma_list:     (OPTIONAL) flattening parameter
    ///                If gamma_list is a vector, one of them is selected by cross validation.

    //cout << "/******************************************/" << endl;
    //cout << "IWLS_train(x,y,sigma_list, lambda_list, gamma_list, b, KR_model0_ytr) " << endl;

    int d = x.rows();//4
    int n = x.cols();
    int ny = y.size();
    int fold = 5; //default parameter

    if( n != ny){
        cout << "sample size of x and y are different!!!" << endl;
    }

    //for now we assume that:
    //-weights are same
    //-Gaussian centers are taken randomly from xtr for now
    Eigen::VectorXd w = VectorXd::Ones(n);

    int sigma_len, gamma_len, lambda_len, iteration_num;
    sigma_len = sigma_list.size();
    gamma_len = gamma_list.size();
    lambda_len = lambda_list.size();
    iteration_num = sigma_len * gamma_len * lambda_len * fold;

    //cout << "d = " << d << endl;
    //cout << "n = " << n << endl;
    //cout << "lambda_list (vector): " << "\n" << lambda_list.transpose() << endl;
    //cout << "sigma_list (vector): " << "\n" << sigma_list.transpose() << endl;
    //cout << "gamma_list (scalar): " << gamma_list.transpose() << endl;

    ///Compute Gaussian kernel centers
    VectorXd rand_index = VectorXd::Zero(n);
    MatrixXd center = MatrixXd::Zero(d, b);
    VectorXd vec1(n), vec2(b);
    MatrixXd mat3(n,b), XX(n,b);

    float min = 1; float max = n;
    for(int i=0; i<n; i++){
       rand_index(i) = (int)( (((float) rand() / (float) RAND_MAX) * (max - min)) + min) ;
    }
     //rand_index << 21,102,55,13,18,41,8,79,17,130,38,6,3,47,75,58,27,7,35,86,19,121,67,1,40,60,2,23,127,123,95,82,120,51,42,39,34,76,56,30,91,111,113,45,105,
      //      29,84,50,114,28,94,44,107,119,103,62,20,54,128,93,90,77,12,125,10,70,83,99,109,37,63,9,53,64,85,65,73,87,92,36,74,101,15,78,115,131,69,48,97,81,
      //      96,124,126,68,57,32,116,88,11,59,31,98,110,132,4,22,14,16,66,43,122,71,5,24,49,104,26,117,46,72,112,52,100,25,129,80,108,33,118,61,106,89;
     //rand_index = rand_index - VectorXd::Ones(n);

    for(int i=0; i<b; i++){
      center.col(i) = x.col( rand_index(i) );
    }

    vec1 = ( x.array()*x.array() ).colwise().sum();         /// sum(x.^2,1)
    vec2 = ( center.array()*center.array() ).colwise().sum(); /// sum(center.^2,1)
    mat3 = -2 * x.transpose() * center;
    XX = vec1.replicate(1,b) + vec2.transpose().replicate(n,1) + mat3; //(x-center)^2
    //cout << "\n" << "XX.size() = " << XX.rows() << " x " << XX.cols() << endl;
    //cout << "XX.row(0): " << "\n" << XX.row(0) << endl; //different in 1.000


    float sigma_chosen, lambda_chosen, gamma_chosen;
    if (sigma_list.size() == 1){
        sigma_chosen = sigma_list(0);
        cout << "sigma_chosen = " << sigma_chosen << endl;
    }
    if (lambda_list.size() == 1){
        lambda_chosen = lambda_list(0);
        cout << "lambda_chosen = " << lambda_chosen << endl;
    }
    if (gamma_list.size() == 1){
        gamma_chosen = gamma_list(0);
        cout << "gamma_chosen = " << gamma_chosen << endl;
    }

    //cout << "center: " << "\n" << center.transpose() << endl;

    ///Searching for Gaussian kernel width "sigma_chosen" and regularization parameter "lambda_chosen"
    ///Using Cross-Validation
    ///*****************************************************Beginning Of Cross-Validation*********************************************************************************/


    MatrixXd score_cv = MatrixXd::Zero(gamma_list.size(), lambda_list.size());
    //cout << "score_cv: " << "\n" << score_cv << endl;

    //cv_index contains n random indices from (1,n)
    //cv_split contains information about this fold and the other folds
    //all elements that belong to this fold are marked with k of this fold
    VectorXd cv_index = VectorXd::Zero(n);
    VectorXd cv_split(n);
    min = 1; max = n;


    for(int i=0; i<n; i++){
       cv_index(i) = (int)( (((float) rand() / (float) RAND_MAX) * (max - min)) + min) ;
    }
    //cv_index << 90,101,69,14,99,1,15,48,5,115,121,83,28,123,122,53,85,47,31,77,98,93,100,55,110,82,75,119,44,39,126,105,7,3,26,106,51,111,
      //      128,56,42,25,10,6,17,130,102,29,37,57,76,22,103,11,87,46,132,41,113,63,62,84,8,2,131,27,107,72,70,54,61,114,65,34,80,43,118,33,
      //      59,13,30,112,36,117,92,88,89,91,23,9,66,104,120,32,95,81,94,86,49,58,71,67,16,24,50,40,21,129,18,4,127,124,96,20,74,60,97,45,78,35,125,52,108,12,19,116,109,79,38,73,68,64;
    //cv_index = cv_index - VectorXd::Ones(n);


    for(int i=0; i<n; i++){
       //cv_index(i) = (int)( (((float) rand() / (float) RAND_MAX) * (max - min)) + min) ;
       cv_split(i) = floor((fold*i)/n) + 1;
    }

    int iteration_idx = 0;

    //cout << "cv_split: " << cv_split << endl;

    ///Cross-validation loop
    VectorXd score_tmp_sigmas = VectorXd::Zero(sigma_len);
    VectorXd lambda_chosen_tmp = VectorXd::Zero(sigma_len);
    VectorXd gamma_chosen_tmp = VectorXd::Zero(sigma_len);



    for(int sigma_index = 0; sigma_index<sigma_len; sigma_index++){
        MatrixXd Ksigma(n,b), aux(n,b);

        float sigma = sigma_list(sigma_index);
        aux = XX/(2*sigma*sigma);

        for(int i=0; i<n; i++){
            for(int j=0; j<b; j++){
                Ksigma(i,j) = exp(-aux(i,j));
            }
        }

        //cout << "sigma = " << sigma << endl;
        //cout << "\n" << "Ksigma.row(0): " << Ksigma.row(0) << endl;

        for(int gamma_index=0; gamma_index<gamma_len; gamma_index++){
            float gamma = gamma_list(gamma_index);
            for(int lambda_index=0; lambda_index<lambda_len; lambda_index++){
                float lambda = lambda_list(lambda_index);
                //cout << "lambda = " << lambda << endl;
                VectorXd score_tmp = VectorXd::Zero(fold); //score for each of 5 folds

                for(int k=0; k<fold; k++){
                    //cout << "k = " << k << endl;
                    ///choose training indices: j=cv_index(cv_split~=k)
                    /// choose test indices
                    /// TODO: to do in a smarter way using vectors or ternary operators or sth like this

                    ///step1: compute #tr, #te points for the k-th fold
                    int numtr = 0;
                    int numte = 0;
                    for(int i=0; i<n;i++){
                        if(cv_split(i) != (k+1)){
                            numtr = numtr+1;
                        }
                    }
                    numte = n - numtr;

                    ///step2: find the indices of te and tr points in Ksigma
                    VectorXd indices_tr(numtr), indices_te(numte);

                    int itr = 0; int ite = 0;

                    for(int i=0; i<n; i++){
                        if(cv_split(i) != (k+1)){
                            //it is a training point for k-fold
                            indices_tr(itr) = cv_index(i);
                            itr = itr + 1;
                        }
                        else{
                            indices_te(ite) = cv_index(i);
                            ite = ite + 1;
                        }
                    }

                    ///step3: create Kcvtr, Kcvte, ycvtr, ycvte
                    MatrixXd Kcvtr(numtr, b), Kcvte(numte, b), Kcv_w(numtr, b);
                    VectorXd ycvtr(numtr), ycvte(numte), aux3(numte);
                    VectorXd wtr(numtr);
                    MatrixXd A(b,b);
                    VectorXd rhs(b), alpha_cv(b);

                    for(int i=0; i<numtr; i++){
                        Kcvtr.row(i) = Ksigma.row( indices_tr(i) );
                        ycvtr(i) = y( indices_tr(i) );
                        //wtr(i) = 1; // 1^0
                        wtr(i) = pow( w(indices_tr(i)), gamma ); //attention: here gamma is only scalar!!
                    }


                    for(int i=0; i<numte; i++){
                        Kcvte.row(i) = Ksigma.row( indices_te(i) );
                        ycvte(i) = y( indices_te(i) );
                    }


                    Kcv_w = Kcvtr.cwiseProduct( wtr.replicate(1,b) );
                    //Kcv_w = Kcvtr;

                    A = Kcvtr.transpose() * Kcv_w + lambda * MatrixXd::Identity(b,b);
                    rhs = Kcv_w.transpose() * ycvtr;
                    //alpha_cv = A.inverse() * rhs;
                    //alpha_cv = A.fullPivLu().solve(rhs);
                    //alpha_cv = A.Jacobian(rhs);
                    alpha_cv = A.lu().solve(rhs); // Stable and fast

                    int dim1 = (ycvte - Kcvte*alpha_cv).cols();
                    int dim2 = (ycvte - Kcvte*alpha_cv).rows();

                    aux3 = (ycvte - Kcvte*alpha_cv).array() * (ycvte - Kcvte*alpha_cv).array();
                    score_tmp(k) = (aux3).mean();


                    if(k==1){
                        //cout << "numte = " << numte << endl;
                        //cout << "cv_index: " << "\n" << cv_index << endl;
                        //cout << "cv_split: " << "\n" << cv_split << endl;
                        //cout << "indices_tr" << "\n" << indices_tr << endl;
                        //cout << "indices_te" << "\n" << indices_te << endl;

                        //cout << "dim1 = " << dim1 << endl;
                        //cout << "dim2 = " << dim2 << endl;
                        //cout << "\n" << "aux3: " << "\n" << aux3 << endl;
                        //cout << score_tmp(k) << endl;
                        //cout << "\n" << "Kcv_w.row(0): " << "\n" << Kcv_w.row(0) << endl;
                        //cout << "rhs: " << "\n" << rhs << endl;
                        //cout << "alpha_cv = " << "\n" << alpha_cv << endl;
                        //float relative_error = (A*alpha_cv - rhs).norm() / rhs.norm(); // norm() is L2 norm
                        //cout << "The relative error is:\n" << relative_error << endl;
                        //cout << "detA = " << A.determinant() << endl;
                        // cout << "score_tmp(k) = " << score_tmp(k) << endl;
                        //cout <<"\n" << "A.row(0): " << "\n" << A.row(0).transpose() << endl;
                        //cout <<"\n" << "ycvte: " << "\n" << ycvte << endl;
                    }


                    iteration_idx = iteration_idx + 1;
                }//k-fold
                //cout << "score_tmp: " << score_tmp << endl;
                score_cv(gamma_index, lambda_index) = score_tmp.mean();

                //cout << "score_cv(gamma_index, lambda_index) = " << score_cv(gamma_index, lambda_index) << endl;
            }//lambda
        }//gamma
            //cout << "score_cv: " << "\n" << score_cv << endl;

//          [score_cv_tmp,lambda_chosen_index]=min(score_cv,[],2);
//          [score_tmp(sigma_index),gamma_chosen_index]=min(score_cv_tmp);
//          lambda_chosen_tmp(sigma_index)=lambda_list(lambda_chosen_index(gamma_chosen_index));
//          gamma_chosen_tmp(sigma_index)=gamma_list(gamma_chosen_index);


            //for now: we use gamma_chosen==0, so score_cv is a matrix with single row
             MatrixXd::Index minIndex;
             float score_cv_tmp =  score_cv.row(0).minCoeff(&minIndex);
             int lambda_chosen_index = minIndex;
             int gamma_chosen_index = 0;

             score_tmp_sigmas(sigma_index) = score_cv_tmp;
             lambda_chosen_tmp(sigma_index) = lambda_list(lambda_chosen_index);
             gamma_chosen_tmp(sigma_index) = gamma_list(gamma_chosen_index);
    }//sigma


    //cout <<"\n" << "score_tmp_sigmas: " << score_tmp_sigmas.transpose() << endl;

    MatrixXd::Index minIndex;
    float score = score_tmp_sigmas.minCoeff(&minIndex);
    //cout << "score = " << score << endl;
    //cout << "minIndex = " << minIndex << endl;
    int sigma_chosen_index = minIndex;
    sigma_chosen_index = 5; /////!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TEMPORARILY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    lambda_chosen = lambda_chosen_tmp(sigma_chosen_index);
    gamma_chosen = gamma_chosen_tmp(sigma_chosen_index);
    sigma_chosen = sigma_list(sigma_chosen_index);

//    [score,sigma_chosen_index]=min(score_tmp);
//    lambda_chosen=lambda_chosen_tmp(sigma_chosen_index);
//    gamma_chosen=gamma_chosen_tmp(sigma_chosen_index);
//    sigma_chosen=sigma_list(sigma_chosen_index);

    //cout << "lambda_chosen = " << lambda_chosen << endl;
    //cout << "sigma_chosen = " << sigma_chosen << endl;
    //cout << "gamma_chosen = " << gamma_chosen << endl;

    ///*****************************************************End Of Cross-Validation*****************************************************************************/
//      K=exp(-XX/(2*sigma_chosen^2));
//      K_w=K.*repmat((w.^gamma_chosen)',[1 b]);
//      model.alpha=mylinsolve(K'*K_w+lambda_chosen*eye(b),K_w'*y');
//      model.sigma=sigma_chosen;
//      model.lambda=lambda_chosen;
//      model.gamma=gamma_chosen;
//      model.center=center;

    MatrixXd K(n,b), aux(n,b), K_w;
    MatrixXd Afinal(b,b);
    VectorXd rhsfinal(b), alpha_cv_final(b);

    aux = XX/(2*sigma_chosen*sigma_chosen);

    for(int i=0; i<n; i++){
        for(int j=0; j<b; j++){
            K(i,j) = exp(-aux(i,j));
        }
    }
    K_w = K.cwiseProduct( w.replicate(1,b) );

    Afinal = K.transpose() * K_w + lambda_chosen * MatrixXd::Identity(b,b);
    rhsfinal = K_w.transpose() * y;
    alpha_cv_final = Afinal.lu().solve(rhsfinal); // Stable and fast

    KR_model0_ytr->alpha = alpha_cv_final;
    KR_model0_ytr->sigma = sigma_chosen;
    KR_model0_ytr->lambda = lambda_chosen;
    KR_model0_ytr->gamma = gamma_chosen;
    KR_model0_ytr->center = center;

    //cout << "/******************************************/" << endl;

}





void kernel_regression_nDmD(Eigen::MatrixXd& xtrain, Eigen::MatrixXd& ytrain, char* opt, KR_model_vs* KR_model_vs_local){
    cout << "/********************************************************/" << endl;
    std::cout <<"\n" << "kernel_regression_nDmD" << std::endl;

    VectorXd xtrain_mean, xtrain_std, ytrain_mean, ytrain_std;

    bool NORMALIZE_DATA = true;
    bool USE_IWLS = false;
    bool REVERSE = false;

    if (REVERSE==1){
        Eigen::MatrixXd tmp = xtrain;
        xtrain = ytrain;
        ytrain = tmp;
    }

    int n = xtrain.cols();
    int d = xtrain.rows();//4
    cout << "xtrain.rows() = " << d << endl;
    cout << "xtrain.cols() = " << n << endl;

    MatrixXd xtr(n,d), ytr(n,d);
    MatrixXd xtrain_normalized(n,d), ytrain_normalized(n,d);


    if(NORMALIZE_DATA == 1){
        xtrain_mean = VectorXd::Zero(d);
        xtrain_std = VectorXd::Zero(d);
        ytrain_mean = VectorXd::Zero(d);
        ytrain_std = VectorXd::Zero(d);

        xtrain_normalized = MatrixXd::Zero(n, d);
        ytrain_normalized = MatrixXd::Zero(n, d);

        for(int i=0; i<d; i++){
            xtrain_mean(i) = xtrain.row(i).mean(); //different in 0.0001 in the last element
            ytrain_mean(i) = ytrain.row(i).mean();
        }

        standDev(xtrain, xtrain_std);
        standDev(ytrain, ytrain_std);

        normalize_dist(xtrain, xtrain_mean, xtrain_std, xtrain_normalized);
        normalize_dist(ytrain, ytrain_mean, ytrain_std, ytrain_normalized);

        std::cout << "xtrain_mean =" << "\n" << xtrain_mean << std::endl;
        std::cout << "\n" << "xtrain_std =" << "\n" << xtrain_std << "\n" << std::endl;
        //std::cout << "\n" << "xtrain_normalized =" << "\n" << xtrain_normalized.transpose() << std::endl; //different in 0.0001

        xtr = xtrain_normalized;
        ytr = ytrain_normalized;

    }
    else{
        xtr = xtrain;
        ytr = ytrain;
    }


    int Nd = ytr.rows(); //Nd=4
    std::cout << "Nd = " << Nd << std::endl;
    //KR_model0_ytr=cell(Nd,1);

    VectorXd lambda_list = VectorXd::Zero(6);
    VectorXd sigma_list;

    if(USE_IWLS == false){
          //Plain Kernel Regularized Least-Squares
          linspace(-9, -1, 6, lambda_list);
          cout << "lambda_list: " << lambda_list.transpose() <<  "\n" << endl;

          int b = std::min(200, n);
          std::cout << "b = " << b << std::endl;
          std::cout << "n = " << n << std::endl;



          VectorXd rand_index = VectorXd::Zero(n);
          float min = 1; float max = n;
          for(int i=0; i<n; i++){
            rand_index(i) = (int)( (((float) rand() / (float) RAND_MAX) * (max - min)) + min) ;
          }
          //rand_index << 69,52,85,91,79,25,74,119,36,118,104,18,108,124,120,132,4,126,45,129,33,113,40,5,86,37,115,32,46,67,51,14,102,7,60,97,121,
            //      82,49,65,106,128,72,43,131,20,105,77,64,84,100,101,48,70,11,12,73,94,93,3,92,8,125,57,42,63,75,89,58,116,10,107,15,87,19,127,62,
              //    130,96,13,21,61,2,122,35,16,109,38,24,39,9,103,26,59,6,88,31,44,117,78,90,41,66,27,95,99,17,1,80,28,98,114,47,81,50,22,110,123,
                //  23,76,54,53,56,111,55,83,71,34,30,68,112,29;
          //rand_index = rand_index - VectorXd::Ones(n);


          /*
           TODO:
            center=xtmp(:,rand_index(1:b));
            XX=repmat(sum(x.^2,1)',[1 b])+repmat(sum(center.^2,1),[n 1])-2*x'*center;
            xscale=sqrt(median(XX(:)));
          */

          MatrixXd center = MatrixXd::Zero(d, b);
          VectorXd vec1(n), vec2(b);
          MatrixXd mat3(n,b);
          MatrixXd XX = MatrixXd::Zero(n,b); //(x-center)^2
          MatrixXd copyXX(n,n);
          VectorXd copyXXvec(n*b);

          for(int i=0; i<b; i++){
            center.col(i) = xtr.col( rand_index(i) );
          }


          ///TODO: to be careful where is n, where is b
          vec1 = ( xtr.array()*xtr.array() ).colwise().sum();         // sum(x.^2,1)
          vec2 = ( center.array()*center.array() ).colwise().sum(); // sum(center.^2,1)
          mat3 = -2 * xtr.transpose() * center; //-2*x'*center
          XX = vec1.replicate(1,b) + vec2.transpose().replicate(n,1) + mat3;

          copyXX = XX;
          copyXX.resize(n*b, 1);
          copyXXvec = copyXX.col(0);


          float median = 0;
          float xscale = 0;
          calcVecMedian(copyXXvec, &median);
          xscale = sqrt(median);

          cout << "\n" << "median = " << median << endl;
          cout << "xscale = " << xscale << endl;

          //if(strcmp(opt, "static_xscale")){
             xscale = 50;
          //}

           VectorXd vec = VectorXd::Zero(9);
           vec << 0.1, 0.2, 0.5, 0.666666667, 1, 1.5, 2, 5, 10;

           sigma_list = xscale * vec;
           VectorXd gamma_list(1);
           gamma_list(0) = 0;
           std::cout << "\n" << "sigma_list: " << "\n" << sigma_list.transpose() << std::endl;


           Nd = 4; //just for now
           for(int i=0; i<Nd; i++){
               cout << i << endl;
               Eigen::VectorXd ytri = ytr.row(i);
               cout << i << endl;
               IWLS_train(xtr,ytri, sigma_list, lambda_list, gamma_list, b, &(KR_model0_ytr[i]) );
           }

           ///TODO: to write a function for printing KR_model0_ytr[i]

           KR_model_vs_local->lambda_list = lambda_list;
           KR_model_vs_local->sigma_list = sigma_list;


    }
    else{
        /*
            %%%%%%%%%%%%%%%%%%%%%%%%% Adaptive Importance-Weighted Kernel Regularized Least-Squares
            [wh_xtr]=uLSIF(xtr,xte); %importance weight
            for d=1:Nd
                KR_model0_ytr{d}=IWLS_train(xtr,ytr(d,:),[],[],sigma_list,lambda_list,0,base); %imp weight is supposed to be given as a parr
            end
        */
    }

    for(int i=0; i<Nd; i++){
        cout << "LS_ytr" << i << ": sigma = " << KR_model0_ytr[i].sigma << ", lambda = " << KR_model0_ytr[i].lambda << endl;
    }

    KR_model_vs_local->xtr = xtr;
    KR_model_vs_local->ytr = ytr;
    if( NORMALIZE_DATA ==1){
        KR_model_vs_local->ytrain_mean = ytrain_mean;
        KR_model_vs_local->ytrain_std = ytrain_std;
        KR_model_vs_local->xtrain_mean = xtrain_mean;
        KR_model_vs_local->xtrain_std = xtrain_std;
    }
    KR_model_vs_local->USE_IWLS = USE_IWLS;
    KR_model_vs_local->NORMALIZE_DATA = NORMALIZE_DATA;
    for(int i=0; i<Nd; i++){
        KR_model_vs_local->KR_model0_ytr[i] = KR_model0_ytr[i];
    }

}








/*
USE_IWLS = Use importance-weighted regularized least
UVST = light field parametrization, i.e. lumigraph


*/

/*
struct KR_model_vs{
    VectorXd lambda_list; //1x6
    VectorXd sigma_list; //1x9
    MatrixXd xtr; //4x132
    MatrixXd ytr; //4x132
    VectorXd ytrain_mean;  //4x1
    VectorXd ytrain_std; //4x1
    VectorXd xtrain_mean; //4x1
    VectorXd xtrain_std; //4x1
    bool USE_IWLS; //0
    bool NORMALIZE_DATA; //1
    KR_model KR_model0_ytr[4]; //4x1 cell
    bool DISTORT_UVST; //1
} KR_model_optical;

*/



void print_KR_model_vs(KR_model_vs* KR_model_optical){
    cout << "\n" << "/****printing KR_model_vs****/ " << endl;
    cout << "lambda_list = " << "\n" << KR_model_optical->lambda_list <<"\n" << endl;
    cout << "sigma_list = " << "\n" << KR_model_optical->sigma_list <<"\n" << endl;
    cout << "ytrain_mean = " << "\n" << KR_model_optical->ytrain_mean <<"\n" << endl;
    cout << "ytrain_std = " << "\n" << KR_model_optical->ytrain_std <<"\n" << endl;
    cout << "xtrain_mean = " << "\n" << KR_model_optical->xtrain_mean <<"\n" << endl;
    cout << "ytrain_std = " << "\n" << KR_model_optical->ytrain_std <<"\n" << endl;
    cout << "USE_IWLS = " << "\n" << KR_model_optical->USE_IWLS <<"\n" << endl;
    cout << "NORMALIZE_DATA = " << "\n" << KR_model_optical->NORMALIZE_DATA <<"\n" << endl;
    cout << "DISTORT_UVST = " << "\n" << KR_model_optical->DISTORT_UVST <<"\n" << endl;


    int Nd = (KR_model_optical->ytr).rows();
    for(int i=0; i<Nd; i++){
        cout << "i = " << i << endl;
        //cout << "KR_model0_ytr[i].alpha = " << "\n" << KR_model_optical->KR_model0_ytr[i].alpha << endl;
        cout << "KR_model0_ytr[i].sigma = " << KR_model_optical->KR_model0_ytr[i].sigma << endl;
        cout << "KR_model0_ytr[i].lambda = " << KR_model_optical->KR_model0_ytr[i].lambda << endl;
        cout << "KR_model0_ytr[i].gamma = " << KR_model_optical->KR_model0_ytr[i].gamma << endl;
        //cout << "KR_model0_ytr[i].center = " << KR_model_optical->KR_model0_ytr[i].center << endl;

    }
    //xtr
    //ytr
}


void output_KR_model_vs(KR_model_vs* KR_model_optical){

     int Nd = 4;
     ofstream myfile;
     myfile.open ("./KR_model_optical/n.txt");
     myfile << KR_model_optical->xtr.cols();
     myfile.close();

     myfile.open ("./KR_model_optical/lambda_list.txt");
     myfile << KR_model_optical->lambda_list;
     myfile.close();

     myfile.open ("./KR_model_optical/sigma_list.txt");
     myfile << KR_model_optical->sigma_list;
     myfile.close();

     myfile.open ("./KR_model_optical/xtr.txt");
     myfile << KR_model_optical->xtr.transpose();
     myfile.close();

     myfile.open ("./KR_model_optical/ytr.txt");
     myfile << KR_model_optical->ytr.transpose();
     myfile.close();

     //myfile << "\n" << "/****printing KR_model_vs****/ " << endl;
    //cout << "lambda_list = " << "\n" << KR_model_optical->lambda_list <<"\n" << endl;
    //cout << "sigma_list = " << "\n" << KR_model_optical->sigma_list <<"\n" << endl;
//     myfile << "ytrain_mean = " << "\n" << KR_model_optical->ytrain_mean.transpose() <<"\n" << endl;
//     myfile << "ytrain_std = " << "\n" << KR_model_optical->ytrain_std.transpose() <<"\n" << endl;
//     myfile << "xtrain_mean = " << "\n" << KR_model_optical->xtrain_mean.transpose() <<"\n" << endl;
//     myfile << "xtrain_std = " << "\n" << KR_model_optical->xtrain_std.transpose() <<"\n" << endl;
    //cout << "USE_IWLS = " << "\n" << KR_model_optical->USE_IWLS <<"\n" << endl;
    // myfile << "NORMALIZE_DATA = " << "\n" << KR_model_optical->NORMALIZE_DATA <<"\n" << endl;
    //cout << "DISTORT_UVST = " << "\n" << KR_model_optical->DISTORT_UVST <<"\n" << endl;
    //int Nd = (KR_model_optical->ytr).rows();



     char d1[50], d2[50], d3[50], d4[50];
     sprintf(d1,"./KR_model_optical/");
     sprintf(d2,"./KR_model_optical/");
     sprintf(d3,"./KR_model_optical/");
     sprintf(d4,"./KR_model_optical/");

    for(int id=1; id<=Nd; id++){      
        char p[50], r[50], s[50];
        sprintf(p,"./KR_model_optical/");
        sprintf(r,"./KR_model_optical/");
        sprintf(s,"./KR_model_optical/");

        char sub[16];
        sprintf(sub, "%d", id);
        strcat(sub, "/");
        strcat(p,sub);
        strcat(r,sub);
        strcat(s, sub);
//        strcat(d1,sub);
//        strcat(d2,sub);
//        strcat(d3, sub);
//        strcat(d4, sub);

        char al[16], cen[16], sig[16];
        sprintf(al, "alpha.txt");
        strcat(p,al);
        sprintf(cen, "center.txt");
        strcat(r,cen);
        sprintf(sig, "sigma.txt");
        strcat(s,sig);
//        sprintf(ym, "ytrain_mean.txt");
//        strcat(d1,ym);
//        sprintf(ys, "ytrain_std.txt");
//        strcat(d2,ys);
//        sprintf(xm, "xtrain_mean.txt");
//        strcat(d3,xm);
//        sprintf(xs, "xtrain_std.txt");
//        strcat(d4,xs);



        ofstream myfile;

       // myfile2.open ("./KR_model_optical/1/alpha.txt");
       // myfile2.open ("./KR_model_optical/1/center.txt");
       // myfile2.open ("./KR_model_optical/1/sigma.txt");
        //cout << id << endl;
        myfile.open(p);
        myfile << KR_model_optical->KR_model0_ytr[id-1].alpha << endl;
        myfile.close();

        myfile.open(r);
        myfile << KR_model_optical->KR_model0_ytr[id-1].center.transpose() << endl;
        myfile.close();

        myfile.open(s);
        myfile << KR_model_optical->KR_model0_ytr[id-1].sigma << endl;
        myfile.close();


    }

    //char d[50];
    //sprintf(d,"./KR_model_optical/");
    char ym[16], ys[16], xm[16], xs[16];

    sprintf(ym, "ytrain_mean.txt");
    strcat(d1,ym);
    sprintf(ys, "ytrain_std.txt");
    strcat(d2,ys);
    sprintf(xm, "xtrain_mean.txt");
    strcat(d3,xm);
    sprintf(xs, "xtrain_std.txt");
    strcat(d4,xs);

    myfile.open(d1);
    myfile << KR_model_optical->ytrain_mean << endl;
    myfile.close();

    myfile.open(d2);
    myfile << KR_model_optical->ytrain_std << endl;
    myfile.close();

    myfile.open(d3);
    myfile << KR_model_optical->xtrain_mean << endl;
    myfile.close();

    myfile.open(d4);
    myfile << KR_model_optical->xtrain_std << endl;
    myfile.close();




}


void IWLS_test(MatrixXd& x, KR_model model, int ind, MatrixXd& ydisph0){
    cout << "IWLS_test in the kernel_regression_apply" << endl;
    //ydisph0.row(i) =
    ///[d,n]=size(x);
    ///b=size(model.center,2);
    ///K=exp(-(repmat(sum(x.^2,1)',[1 b])+repmat(sum(model.center.^2,1),[n 1])...
     ///       -2*x'*model.center)/(2*model.sigma^2));
    ///y=(K*model.alpha)';

    int d = x.rows();
    int n = x.cols();//224
    int b = model.center.cols(); //100

    VectorXd vec1(n), vec2(b), y(n);
    MatrixXd mat1(n,b), mat2(n,b), mat3(n,b);

    vec1 = (x.array()*x.array()).colwise().sum(); //different in 0.02
    vec2 = (model.center.array()*model.center.array()).colwise().sum();

    mat1 = vec1.replicate(1,b) + vec2.transpose().replicate(n,1);
    mat2 = -2 * x.transpose()* model.center;
    mat3 = -1*(mat1+mat2)/(2*model.sigma*model.sigma);

    MatrixXd K(n,b);
    for(int i=0; i<n; i++){
        for(int j=0; j<b; j++){
            K(i,j) = exp(mat3(i,j));
        }
    }

    y = K*model.alpha;
    //cout << y << endl;

    ydisph0.row(ind) = y;

}


void kernel_regression_apply(MatrixXd& xtest, KR_model_vs KR_model_vs_local, MatrixXd& yte){
    cout << "\n" << "kernel_regression_apply(xtest, KR_model_optical, yte)" << endl;

    //bool NORMALIZE_DATA = KR_model_vs_local.NORMALIZE_DATA;
    bool NORMALIZE_DATA = true;
    //int length = KR_model_vs_local.xtr.cols();
    //cout << "length = " << length << endl;
    KR_model KR_model0_ytr[4] = KR_model_vs_local.KR_model0_ytr;

    VectorXd xtrain_mean(4), xtrain_std(4), ytrain_mean(4), ytrain_std(4);
    MatrixXd xte(xtest.rows(), xtest.cols());

    if(NORMALIZE_DATA){
       xtrain_mean = KR_model_vs_local.xtrain_mean;
       cout << xtrain_mean << endl;
       xtrain_std = KR_model_vs_local.xtrain_std;
       ytrain_mean = KR_model_vs_local.ytrain_mean;
       ytrain_std = KR_model_vs_local.ytrain_std;

       MatrixXd xtest_normalized(xtest.rows(), xtest.cols());
       normalize_dist(xtest, xtrain_mean, xtrain_std, xtest_normalized);
       xte = xtest_normalized;
    }
    else{
        xte = xtest;
    }

    int Nd = 4; //KR_model_vs_local.xtr.rows();
    int Nsample = xtest.cols();
    cout << "Nsample = " << Nsample << endl;
    MatrixXd ydisph0 = MatrixXd::Zero(Nd, Nsample);

    for(int i=0; i<Nd; i++){
        //ydisph0.row(i) =
        cout << i << endl;
        IWLS_test(xte, KR_model0_ytr[i], i, ydisph0);
    }


     MatrixXd ytest_normalized(Nd, Nsample);

     if(NORMALIZE_DATA){
         ytest_normalized = ydisph0;
         unnormalize_dist(ytest_normalized, ytrain_mean, ytrain_std, yte);
     }
     else{
         yte = ydisph0;
     }

     //cout << yte << endl;

}


void read_KR_model_vs_fromFiles(KR_model_vs* KR_model_optical_fromFile){

    int n = 132 + 2*44;
    int Nd = 4;

    ifstream myReadFile;
    cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;


     myReadFile.open("./KR_model_optical/lambda_list.txt");
     VectorXd lambda_list = VectorXd::Zero(6);
     if (myReadFile.is_open()) {
         float a, b, c, d, e, f;
         myReadFile >> a >> b >> c >> d >> e >> f ;
         lambda_list(0) = a;
         lambda_list(1) = b;
         lambda_list(2) = c;
         lambda_list(3) = d;
         lambda_list(4) = e;
         lambda_list(5) = f;
         //cout << "lambda_list read from file: " << endl;
         //cout << "\n" << lambda_list << endl;
    }
    myReadFile.close();

    myReadFile.open("./KR_model_optical/sigma_list.txt");
    VectorXd sigma_list = VectorXd::Zero(9);
    if (myReadFile.is_open()) {
        float a, b, c, d, e, f, g, h, k;
        myReadFile >> a >> b >> c >> d >> e >> f  >> g >> h >> k;
        sigma_list(0) = a;
        sigma_list(1) = b;
        sigma_list(2) = c;
        sigma_list(3) = d;
        sigma_list(4) = e;
        sigma_list(5) = f;
        sigma_list(6) = g;
        sigma_list(7) = h;
        sigma_list(8) = k;
        //cout << "sigma_list read from file: " << endl;
        //cout << "\n" << sigma_list << endl;
   }
   myReadFile.close();


        myReadFile.open("./KR_model_optical/xtrain_mean.txt");
        int dim = 4;
        VectorXd xtrain_mean = VectorXd::Zero(dim);
        if (myReadFile.is_open()) {
            float a, b, c, d;
            myReadFile >> a >> b >> c >> d;
            xtrain_mean(0) = a;
            xtrain_mean(1) = b;
            xtrain_mean(2) = c;
            xtrain_mean(3) = d;
            cout << "\n" << xtrain_mean << endl;
       }
       myReadFile.close();

       myReadFile.open("./KR_model_optical/xtrain_std.txt");
       VectorXd xtrain_std = VectorXd::Zero(dim);
       if (myReadFile.is_open()) {
           float a, b, c, d;
           myReadFile >> a >> b >> c >> d;
           xtrain_std(0) = a;
           xtrain_std(1) = b;
           xtrain_std(2) = c;
           xtrain_std(3) = d;
           cout << "\n" << xtrain_std << endl;
      }
      myReadFile.close();


      myReadFile.open("./KR_model_optical/ytrain_mean.txt");
      VectorXd ytrain_mean = VectorXd::Zero(dim);
      if (myReadFile.is_open()) {
          float a, b, c, d;
          myReadFile >> a >> b >> c >> d;
          ytrain_mean(0) = a;
          ytrain_mean(1) = b;
          ytrain_mean(2) = c;
          ytrain_mean(3) = d;
          //cout << "\n" << ytrain_mean << endl;
     }
     myReadFile.close();


     myReadFile.open("./KR_model_optical/ytrain_std.txt");
     VectorXd ytrain_std = VectorXd::Zero(dim);
     if (myReadFile.is_open()) {
         float a, b, c, d;
         myReadFile >> a >> b >> c >> d;
         ytrain_std(0) = a;
         ytrain_std(1) = b;
         ytrain_std(2) = c;
         ytrain_std(3) = d;
         //cout << "\n" << ytrain_std << endl;
    }
    myReadFile.close();


    myReadFile.open("./KR_model_optical/xtr.txt");
    MatrixXd xtrr = MatrixXd::Zero(n,4); ///TODO!!!!!!!!!!!!!! to change the dimensions to the  (4,n) compare printing with other model!!!!!!!!
    MatrixXd xtr = MatrixXd::Zero(4,n);

    if(myReadFile.is_open()){
        for(int i=0; i<n; i++){
            float a,b,c,d;
            myReadFile >> a >> b >> c >> d;
            xtrr(i,0) = a;
            xtrr(i,1) = b;
            xtrr(i,2) = c;
            xtrr(i,3) = d;

            if(i==n-1){
                cout << "xtr(n-1): " << a << ", " << b << ", " << c << ", " << d << endl;
            }

        }
        xtr = xtrr.transpose();

    }
    else{
        cout << "error reading file" << endl;
    }
    myReadFile.close();

    myReadFile.open("./KR_model_optical/ytr.txt");
    MatrixXd ytrr = MatrixXd::Zero(n,4);
    MatrixXd ytr = MatrixXd::Zero(4,n);
    if(myReadFile.is_open()){
        for(int i=0; i<n; i++){
            float a,b,c,d;
            myReadFile >> a >> b >> c >> d;
            ytrr(i,0) = a;
            ytrr(i,1) = b;
            ytrr(i,2) = c;
            ytrr(i,3) = d;

            if(i==n-1){
                cout << "ytr(n-1): " << a << ", " << b << ", " << c << ", " << d << endl;
            }
        }
        ytr = ytrr.transpose();
    }
    else{
        cout << "error reading file" << endl;
    }
    myReadFile.close();





    for(int id=1; id<=Nd; id++){
        char p[50], r[50], sub[50], al[20], cen[20];
        sprintf(p, "./KR_model_optical/");
        sprintf(r, "./KR_model_optical/");
        sprintf(sub, "%d", id);
        strcat(sub, "/");
        sprintf(al, "alpha.txt");
        sprintf(cen, "center.txt");
        strcat(p, sub);
        strcat(r, sub);
        strcat(p,al);
        strcat(r, cen);
        //cout << p << endl;
        //cout << r << endl;

        ifstream ifile;

        ifile.open(p);
        VectorXd alpha = VectorXd::Zero(n);
        if(ifile.is_open()){
            for(int i=0; i<n; i++){
                float a;
                ifile >> a;
                alpha(i) = a;
            }
        }
        else{
            cout << "error reading file" << endl;
        }
        ifile.close();


        ifile.open(r);
        MatrixXd centerr = MatrixXd::Zero(n,4);
        MatrixXd center = MatrixXd::Zero(4,n);
        if(ifile.is_open()){
            for(int i=0; i<n; i++){
                float a,b,c,d;
                ifile >> a >> b >> c >> d;
                centerr(i,0) = a;
                centerr(i,1) = b;
                centerr(i,2) = c;
                centerr(i,3) = d;
                if(i==n-1){
                 //   cout << a << ", " << b << ", " << c << ", " << d << endl;
                }
            }
            center = centerr.transpose();
        }
        else{
            cout << "error reading file" << endl;
        }
        ifile.close();




        KR_model_optical_fromFile->KR_model0_ytr[id-1].alpha = alpha;
        KR_model_optical_fromFile->KR_model0_ytr[id-1].center =  center;
        KR_model_optical_fromFile->KR_model0_ytr[id-1].sigma = 75;
        KR_model_optical_fromFile->KR_model0_ytr[id-1].gamma = 0;
        KR_model_optical_fromFile->KR_model0_ytr[id-1].lambda = 0.0;

    }




    KR_model_optical_fromFile->xtrain_mean = xtrain_mean;
    KR_model_optical_fromFile->xtrain_std = xtrain_std;
    KR_model_optical_fromFile->ytrain_mean = ytrain_mean;
    KR_model_optical_fromFile->ytrain_std = ytrain_std;
    KR_model_optical_fromFile->NORMALIZE_DATA = true;
    KR_model_optical_fromFile->DISTORT_UVST =true;
    KR_model_optical_fromFile->USE_IWLS = false;

    KR_model_optical_fromFile->lambda_list = lambda_list;
    KR_model_optical_fromFile->sigma_list = sigma_list;


    KR_model_optical_fromFile->xtr = xtr;
    KR_model_optical_fromFile->ytr = ytr;




}


void computeDistortionMap(KR_model_vs KR_model_optical, Matrix3d& R_E0W, Vector3d& t_E0W,  float offsetw, float offseth, float meter2pixel,  Matrix3d& R_SW_eig, Vector3d& t_SW_eig, Vector3d& t_SW_z0_eig, Matrix3d& K_E_eig){
        Matrix3d R_WE_test;
        R_WE_test = R_E0W.transpose();

        Vector3d t_WE_test;
        t_WE_test = - R_E0W.transpose() * t_E0W;


       /// grid_step=50;
       ///%[u,v] = meshgrid(1:grid_step:1280,1:grid_step:1024);%%% IEEE VR video
       /// [u,v] = meshgrid(300:grid_step:1080,150:grid_step:800);%%%
       /// uv=[u(:) v(:)]';


        /*******************************************************************************************************/
        int grid_step = 50;
        int x0, xend, xnum, y0, yend, ynum, xendnew, yendnew;
        x0 = 1;
        xend =1280;
        y0 = 1;
        yend = 1024;
        xnum = (int)floor((xend-x0)/grid_step) + 1; //number of grid points in x direction
        ynum = (int) floor((yend-y0)/grid_step) + 1;
        xendnew = x0 + (xnum-1)*grid_step;
        yendnew = y0 + (ynum-1)*grid_step;
        VectorXi x = VectorXi::LinSpaced(Sequential,xnum,x0,xendnew);
        VectorXi y = VectorXi::LinSpaced(Sequential, ynum, y0, yendnew);
        //MatrixXi u = x.transpose().replicate(ynum,1);
        //MatrixXi v = y.replicate(1,xnum);

        int uvlength = (xnum)*(ynum);
        cout << "uvlength = " << uvlength << endl;
        MatrixXi uv = MatrixXi::Zero(2, uvlength); //2 rows, unum cols
        MatrixXi uvdist = MatrixXi::Zero(2, uvlength); //2 rows, unum cols

        for(int i=0; i<xnum; i++){
            for(int j=0; j<ynum; j++){
                int ind = j + i*(ynum);
                uv(0, ind) = x(i);
                uv(1,ind) = y(j);
            }
        }


       /// %gt - ground truth, 3D points in the world (eye coord system) corresponding to each pixel,
       /// %the position on the line doesnt matter, cant get the depth infn
       /// xyz_gt = R_WE'*(K_E\[uv;ones(1,size(uv,2))]) + repmat(-R_WE'*t_WE,1,size(uv,2));
       /// %convert to the world coord system
       /// Y0=repmat(-R_WE'*t_WE,1,size(xyz_gt,2)); % t_EW

        MatrixXd matrix(3, uvlength);
        matrix<< uv.cast<double>(), MatrixXd::Ones(1, uvlength);
        //MatrixXd matrix2 = MatrixXd::Zero(3, uvlength);
        //matrix2 = K_E_eig.lu().solve(matrix);

        Vector3d vec;
        MatrixXd xyz_gt(3, uvlength), Y0test(3,uvlength);
        MatrixXd uv_S2 = MatrixXd::Zero(2, uvlength);
        MatrixXd uvdouble(2,uvlength);
        MatrixXd X0test(3,uvlength);
        MatrixXd XS_W0_test(3,uvlength), XS_S_test(3,uvlength), aux(3,uvlength), UV_test(3,uvlength);
        MatrixXd xtest(4, uvlength), yte(4, uvlength);

        vec = -R_WE_test.transpose()*t_WE_test;
        //cout << vec << endl;
        xyz_gt = R_WE_test.transpose()*(K_E_eig.lu().solve(matrix)) + vec.replicate(1, uvlength);
        //cout << xyz_gt << endl;
        Y0test = vec.replicate(1, uvlength); //t_EW

        ///% dataset for uvst
        ///uv_S2(1,:)=(uv(1,:)-offsetw)/meter2pixel;
        ///uv_S2(2,:)=(uv(2,:)-offseth)/meter2pixel;
        ///X0 = [uv_S2;zeros(1,length(uv))];
        ///X0 = R_SW*X0 + repmat(t_SW,1,size(X0,2));
        ///[XS_W0,XS_S] = intersetRayWithPlane(X0,Y0,t_SW,   R_SW);
        ///[~,UV_test]  = intersetRayWithPlane(X0,Y0,t_SW_z0,R_SW);
        ///xtest = [UV_test(1:2,:);XS_S(1:2,:)];
        ///yte = kernel_regression_apply(xtest,KR_model_optical); %red points on the pic.


        uvdouble = uv.cast<double>();
        uv_S2.row(0) = (uvdouble.row(0) - offsetw*VectorXd::Ones(uvlength).transpose()) / meter2pixel;
        uv_S2.row(1) = (uvdouble.row(1) - offseth*VectorXd::Ones(uvlength).transpose()) / meter2pixel;
        //cout << uv_S2 << endl;

        X0test << uv_S2, MatrixXd::Zero(1, uvlength);
        X0test = R_SW_eig * X0test + t_SW_eig.replicate(1, uvlength);
        //cout << X0test << endl;
        /***************************************************************************************************************/


        intersectRayWithPlane(X0test, Y0test, t_SW_eig, R_SW_eig, XS_W0_test, XS_S_test);
        intersectRayWithPlane(X0test, Y0test, t_SW_z0_eig, R_SW_eig, aux, UV_test);

        xtest << UV_test.row(0), UV_test.row(1), XS_S_test.row(0), XS_S_test.row(1);
        cout <<"xtest.col(0): \n" << xtest.col(0) <<  endl;
        cout <<"xtest.col(545): \n" << xtest.col(545) << endl;

        kernel_regression_apply(xtest, KR_model_optical, yte);
        cout << "yte.col(0): \n: "<< yte.col(0) << endl;
        cout << "yte.col(545): \n: "<< yte.col(545) << endl;


    //    %to compute distorted pixel point
    //    uvdist = uv;
    //    uvdist(1, :) = meter2pixel * yte(3,:) + offsetw;
    //    uvdist(2, :) = meter2pixel * yte(4,:) + offseth;

    //    disp('uv(:,1)')
    //    uv(:,1)
    //    disp('uv_S2(:,1)')
    //    uv_S2(:,1)
    //    disp('xtest(:,1)')
    //    xtest(:,1)
    //    disp('yte(:,1)')
    //    yte(:,1)
    //    disp('uvdist(:,1)')
    //    uvdist(:,1)

        MatrixXd yte_m2p = yte * meter2pixel;
        for(int i = 0; i<uvlength; i++){
            uvdist(0,i) =  yte_m2p(2,i) + offsetw;
            uvdist(1,i) =  yte_m2p(3,i) + offseth;
        }

        cout << " uv.col(0): \n" << uv.col(1) << "\n" <<  endl;
        cout << " xtest.col(0): \n" << xtest.col(1) << "\n" <<  endl;
        cout << " yte.col(0): \n" << yte.col(1) << "\n" <<  endl;
        cout << " uvdist.col(0): \n" << uvdist.col(1) << "\n" <<  endl;
        cout << "hello" << endl;


        MatrixXi dist = MatrixXi::Zero(2, uvlength);
        dist.row(0) = uv.row(0) - uvdist.row(0);
        dist.row(1) = uv.row(1) - uvdist.row(1);

        cout << " dist.col(0): \n" << dist.col(500) << "\n" <<  endl;
}




int main(){

    MatrixXd UVST_eig, UVST_distorted_eig;
    MatrixXd XS_W_eig, XS_S_eig,  XS_W_distorted_eig, XS_S_distorted_eig;
    MatrixXd X_eig, Y_eig,   X_distorted_eig, G_E_eig, G_distorted_E_eig;
    MatrixXd t_E0W_eig;

    double meter2pixel;
    Vector3d t_SW_eig;
    Matrix3d R_SW_eig = Matrix3d::Zero();
    Vector3d t_SW_z0_eig = Vector3d::Zero();
    Vector3d t_WS_eig;
    Matrix3d R_WS_eig = Matrix3d::Zero();


    /// file_K = './data/cameraintrinsics.txt'
    /// file_screen = './data/virtual_screen'
    /// root = 'data/'

	int N_id = 17;
	int w_userview = 1280;
	int h_userview = 1024;
    //bool IGNORE_EYE_POSITION = true;
    //bool NORMALIZE_DATA = true;
    //bool USE_IWLS = false;
	bool DISTORT_UVST = true;

    //float offsetw = 0;
    //float offseth = 0;

    ///K_raw = openUbitrack3x3MatrixCalib(file_K);
    float K_raw[3][3] = {
    {+1575, 0, -664.3484},
    {0, +1570.5, -513.3597 },
    { 0, 0, -1 }
	};

    ///K_E = convert_intrinsic_matrix2OpenGL(K_raw, h_userview);
    float K_E[3][3] = {
	{1575, 0, -664.3},
	{0, -1570.5, -509.6 },
	{0, 0, -1 }
	};

    Matrix3d K_E_eig = Matrix3d::Zero();
    K_E_eig << 1575,  0, -664.3,
               0, -1570.5, -509.6 ,
               0,   0,      -1;

	char* root;
	root = "./data/";

    ///file_screen = [kRoot,'data/virtual_screen'];%alpha = 1950.9, R_SW, t_SW, 1280x1024
    char file_screen[60] = "";
    strcat(file_screen, root);
    strcat(file_screen, "virtual_screen");
    std::cout <<"\n" << file_screen << "\n" << std::endl;

    misc.quiverscale=1;
    misc.h = h_userview;


    N_id = 17;
    for(int id=1; id<=N_id;id++){
        std::cout << "\n" << id << std::endl;
		char sub[16];
        sprintf(sub, "%d", id);
		strcat(sub, "/");	
		char dir[16]="";
		strcat(dir, root);
		strcat(dir, sub);			

        //allocate memory for arrays
        int size = 44;

        MatrixXd X0_eig = MatrixXd::Zero(3,size);
        MatrixXd Y0_eig = MatrixXd::Zero(3,size);
        MatrixXd X_distorted0_eig = MatrixXd::Zero(3,size);
        MatrixXd G_E0_eig = MatrixXd::Zero(3,size);
        MatrixXd G_distorted_E0_eig = MatrixXd::Zero(3,size);
        Vector3d t_E0W0_eig, t_W0E0_eig;

        MatrixXd XS_W0_eig = MatrixXd::Zero(3,size);
        MatrixXd XS_S0_eig = MatrixXd::Zero(3,size);
        MatrixXd XS_W_distorted0_eig = MatrixXd::Zero(3,size);
        MatrixXd XS_S_distorted0_eig = MatrixXd::Zero(3,size);

        MatrixXd UV_3N_eig = MatrixXd::Zero(3, size);
        MatrixXd UV_distorted_3N_eig = MatrixXd::Zero(3, size);
        MatrixXd aux_eig = MatrixXd::Zero(3, size);

        MatrixXd UV_eig = MatrixXd::Zero(2, size);
        MatrixXd UV_distorted_eig = MatrixXd::Zero(2, size);
        MatrixXd ST_eig = MatrixXd::Zero(2,size);
        MatrixXd ST_distorted_eig = MatrixXd::Zero(2,size);
        MatrixXd UVST0_eig = MatrixXd::Zero(4,size);
        MatrixXd UVST_distorted0_eig = MatrixXd::Zero(4,size);



        int N; //number of points in data sets
        loadDataSetEigen(dir, K_E, misc, &N, X0_eig, Y0_eig, X_distorted0_eig, t_E0W0_eig, t_W0E0_eig, G_E0_eig, G_distorted_E0_eig);
        //std::cout << "Here is the matrix X0_eig: " << "\n" << X0_eig.transpose() << std::endl;
        //std::cout << "Here is the matrix Y0_eig: " << "\n" << Y0_eig.transpose() << std::endl;
        //std::cout << "Here is the matrix G_E0_eig: " << "\n" << G_E0_eig.transpose() << std::endl;
        //std::cout << "Here is the matrix G_distorted_E0_eig " << "\n" << G_distorted_E0_eig.transpose() << std::endl;
        //std::cout << "Here is the vector t_E0W0_eig: " << "\n" << t_E0W0_eig << "\n" << std::endl;
        //std::cout << "Here is the vector t_E0W0_eig: " << "\n" << t_W0E0_eig << std::endl;

        /*
        tmp=load(file_screen);
        //use newly estimated virtual screen plane
        R_SW        = tmp.R_SW
        t_SW        = tmp.t_SW
        meter2pixel = tmp.alpha

        %R_SW=R_WS';
        %t_SW=-R_WS'*t_WS;
        %t_SW_c=-R_WS'*t_WS_c;
        tmp=load(file_screen);
        if 1 % use newly estimated virtual screen plane
            R_SW        = tmp.R_SW
            %t_SW        = tmp.t_SW_c;
            t_SW        = tmp.t_SW
            meter2pixel = tmp.alpha
        end
        */

        //BE CAREFUL WHICH meter2pixel and P_SW parameters we are using


         //data set 1 for Epson HMD
        /*
        meter2pixel = 1950.9;

        t_SW_eig(0) =  -0.3696; t_SW_eig(1) = -0.3621; t_SW_eig(2) =  -0.6137;

        R_SW_eig << 0.9996,   -0.0237,    0.0150,
                    0.0240,    0.9996,   -0.0179,
                   -0.0145,    0.0183,    0.9997;
        if(id==1){
            cout << "meter2pixel = " << meter2pixel << endl;
            std::cout << "t_SW_eig: \n " << t_SW_eig << std::endl;
            std::cout << "R_SW_eig: \n " << R_SW_eig << std::endl;
         }

        */


        //data set 2

        meter2pixel = 1656.5;
        t_WS_eig(0) =  0.0485; t_WS_eig(1) = 0.1055; t_WS_eig(2) =  0.7243;

        R_WS_eig << 0.9997,   0.0242,  -0.0083,
                    -0.0240,  0.9996, 0.0169,
                    0.0087,   -0.0167,   0.9998;
        if(id==1){
            cout << "meter2pixel = " << meter2pixel << endl;
            std::cout << "t_WS_eig: \n " << t_WS_eig << std::endl;
            std::cout << "R_WS_eig: \n " << R_WS_eig << std::endl;
         }


        R_SW_eig = R_WS_eig.transpose();
        t_SW_eig = - R_WS_eig.transpose() * t_WS_eig;

        if(id==1){
            cout << "meter2pixel = " << meter2pixel << endl;
            std::cout << "t_SW_eig: \n " << t_SW_eig << std::endl;
            std::cout << "R_SW_eig: \n " << R_SW_eig << std::endl;
         }





        //find the intersections of light rays and the virtual display
        intersectRayWithPlane(X0_eig, Y0_eig, t_SW_eig, R_SW_eig, XS_W0_eig, XS_S0_eig);
        //for X_distorted0
        intersectRayWithPlane(X_distorted0_eig, Y0_eig, t_SW_eig, R_SW_eig, XS_W_distorted0_eig, XS_S_distorted0_eig);
        //std::cout << "XS_W_distorted0_eig: \n" << XS_W_distorted0_eig.transpose() << std::endl;
        //std::cout << "XS_S_distorted0_eig: \n" << XS_S_distorted0_eig.transpose() << std::endl;


        t_SW_z0_eig = t_SW_eig;
        t_SW_z0_eig(2) = 0;
        //find the intersection of light rays and the virtual display
        intersectRayWithPlane(X0_eig, Y0_eig, t_SW_z0_eig, R_SW_eig, aux_eig, UV_3N_eig);
        //find the intersection of light rays and the virtual display for X_distorted0
        intersectRayWithPlane(X_distorted0_eig, Y0_eig, t_SW_z0_eig, R_SW_eig, aux_eig, UV_distorted_3N_eig);

        UV_eig = UV_3N_eig.block(0,0,2,size);
        UV_distorted_eig = UV_distorted_3N_eig.block(0,0, 2,size);
        ST_eig = XS_S0_eig.block(0,0, 2,size);
        ST_distorted_eig = XS_S_distorted0_eig.block(0,0, 2,size);

        UVST0_eig << UV_eig, ST_eig; //eq.6
        UVST_distorted0_eig << UV_distorted_eig, ST_distorted_eig; //eq.6

         //append data from the new viewpoint
        MatrixXd old_eig, olddist_eig;
        if(id==1){
            old_eig = MatrixXd::Zero(4, 0);
            olddist_eig = MatrixXd::Zero(4, 0);
        }
        else{
            old_eig = UVST_eig;
            olddist_eig = UVST_distorted_eig;
        }

        UVST_eig.resize(4, old_eig.cols() + UVST0_eig.cols() );
        UVST_distorted_eig.resize(4, olddist_eig.cols() + UVST_distorted0_eig.cols() );
        UVST_eig << old_eig, UVST0_eig;
        UVST_distorted_eig << olddist_eig, UVST_distorted0_eig;



        /*
         %% append data
            XS_W=[XS_W XS_W0];
            XS_S=[XS_S XS_S0];
            XS_W_distorted=[XS_W_distorted XS_W_distorted0];
            XS_S_distorted=[XS_S_distorted XS_S_distorted0];
            X=[X X0];
            Y=[Y Y0];           

            X_distorted=[X_distorted X_distorted0];
            G_E=[G_E G_E0];
            G_distorted_E=[G_distorted_E G_distorted_E0];

            t_E0W=[t_E0W repmat(t_E0W0,1,size(X0,2))];
        */




        MatrixXd old_xsw_eig, old_xss_eig, old_xswdist_eig, old_xssdist_eig;
        MatrixXd old_x_eig, old_y_eig;
        MatrixXd old_xdist_eig, old_ge_eig, old_gdiste_eig;
        MatrixXd old_te0w_eig;

        if(id==1){
            old_xsw_eig = MatrixXd::Zero(3,0);
            old_xss_eig = MatrixXd::Zero(3,0);
            old_xswdist_eig = MatrixXd::Zero(3,0);
            old_xssdist_eig = MatrixXd::Zero(3,0);
            old_x_eig = MatrixXd::Zero(3,0);
            old_y_eig = MatrixXd::Zero(3,0);
            old_xdist_eig = MatrixXd::Zero(3,0);
            old_ge_eig = MatrixXd::Zero(3,0);
            old_gdiste_eig = MatrixXd::Zero(3,0);
            old_te0w_eig = MatrixXd::Zero(3,0);
        }
        else{
            old_xsw_eig = XS_W_eig;
            old_xss_eig = XS_S_eig;
            old_xswdist_eig = XS_W_distorted_eig;
            old_xssdist_eig = XS_S_distorted_eig;
            old_x_eig = X_eig;
            old_y_eig = Y_eig;
            old_xdist_eig = X_distorted_eig;
            old_ge_eig = G_E_eig;
            old_gdiste_eig = G_distorted_E_eig;
            old_te0w_eig = t_E0W_eig;
        }

        XS_W_eig.resize(3,old_xsw_eig.cols() + XS_W0_eig.cols() );
        XS_S_eig.resize(3,old_xss_eig.cols() + XS_S0_eig.cols() );
        XS_W_distorted_eig.resize(3, old_xswdist_eig.cols() + XS_W_distorted0_eig.cols());
        XS_S_distorted_eig.resize(3,old_xssdist_eig.cols() + XS_S_distorted0_eig.cols());
        X_eig.resize(3, old_x_eig.cols() + X0_eig.cols() );
        Y_eig.resize(3, old_y_eig.cols() + Y0_eig.cols() );
        X_distorted_eig.resize(3, old_xdist_eig.cols() + X_distorted0_eig.cols() );
        G_E_eig.resize(3, old_ge_eig.cols() + G_E0_eig.cols() );
        G_distorted_E_eig.resize( 3, old_gdiste_eig.cols() + G_distorted_E0_eig.cols() );
        t_E0W_eig.resize(3, old_te0w_eig.cols() + size*t_E0W0_eig.cols() );

        XS_W_eig << old_xsw_eig, XS_W0_eig;
        XS_S_eig << old_xss_eig, XS_S0_eig;
        XS_W_distorted_eig << old_xswdist_eig, XS_W_distorted0_eig;
        XS_S_distorted_eig << old_xssdist_eig, XS_S_distorted0_eig;
        X_eig << old_x_eig, X0_eig;
        Y_eig << old_y_eig, Y0_eig;
        X_distorted_eig << old_xdist_eig, X_distorted0_eig;
        G_E_eig << old_ge_eig, G_E0_eig;
        G_distorted_E_eig << old_gdiste_eig, G_distorted_E0_eig;


        t_E0W_eig << old_te0w_eig, t_E0W0_eig.replicate(1,size);

        //free memory


    }//iterations over id (of datasets)



    Eigen::MatrixXd xtrain = UVST_eig;
    Eigen::MatrixXd ytrain = UVST_distorted_eig;
    char opt[20] = "static_xscale";
    //cout << ytrain.row(0)<<endl;


    kernel_regression_nDmD(xtrain, ytrain, opt, &KR_model_optical);
    KR_model_optical.DISTORT_UVST = DISTORT_UVST;
    print_KR_model_vs(&KR_model_optical);
    output_KR_model_vs(&KR_model_optical);


    /*****************************************************************************************************/
    //INPUT: KR_model_optical, R_EOW, t_EOW, offsetw, offseth, meter2pixel, K_E_eig
    //OUTPUT: mapping between uv and uvdist

//    Matrix3d R_E0W_test = Matrix3d::Zero();
//    R_E0W_test <<    0.996226308432124,   0.031066683714960,  -0.081043220263370,
//                -0.028404458041297,   0.999024955127270,   0.033798310554464,
//                 0.082014200911078,  -0.031368777405414,   0.996137375392073;

//    Vector3d t_E0W_test = Vector3d::Zero();
//    t_E0W_test <<  -0.029793673250077, -0.078804286612701, 0.118703726732110;


    //read KR_model from set of files in the folder
    //KR_model_vs KR_model_optical_fromFile;
    //read_KR_model_vs_fromFiles(&KR_model_optical_fromFile);
    //print_KR_model_vs(&KR_model_optical_fromFile);

    //computeDistortionMap(KR_model_optical_fromFile, R_E0W_test, t_E0W_test,  offsetw,  offseth,  meter2pixel, R_SW_eig, t_SW_eig, t_SW_z0_eig, K_E_eig);





	return 0;
}

