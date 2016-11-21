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



namespace Eigen{
    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix){
        std::ofstream out(filename,ios::out | ios::binary | ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        //out.write((char*) (&rows), sizeof(typename Matrix::Index));
        //out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }
    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix){
        std::ifstream in(filename,ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }
}



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


void read_KR_model_vs_fromFiles(KR_model_vs* KR_model_optical_fromFile){

    int n = 132; //number of points in a training dataset
    int dim = 4;

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
    else{
         cout << "error reading file" << endl;
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
   else{
        cout << "error reading file" << endl;
   }
   myReadFile.close();


   myReadFile.open("./KR_model_optical/xtrain_mean.txt");
   VectorXd xtrain_mean = VectorXd::Zero(dim);
      if (myReadFile.is_open()) {
          float a, b, c, d;
          myReadFile >> a >> b >> c >> d;
          xtrain_mean(0) = a;
          xtrain_mean(1) = b;
          xtrain_mean(2) = c;
          xtrain_mean(3) = d;
          //cout << "\n" << xtrain_mean << endl;
   }
   else{
         cout << "error reading file" << endl;
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
         //cout << "\n" << xtrain_std << endl;
   }
   else{
       cout << "error reading file" << endl;
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
   else{
       cout << "error reading file" << endl;
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
   else{
       cout << "error reading file" << endl;
   }
   myReadFile.close();


    myReadFile.open("./KR_model_optical/xtr.txt");
    MatrixXd xtrr = MatrixXd::Zero(n,4);
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





    for(int id=1; id<=dim; id++){
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


void IWLS_test(MatrixXd& x, KR_model model, int ind, MatrixXd& ydisph0){
    //cout << "IWLS_test in the kernel_regression_apply" << endl;
    //ydisph0.row(i) =
    ///[d,n]=size(x);
    ///b=size(model.center,2);
    ///K=exp(-(repmat(sum(x.^2,1)',[1 b])+repmat(sum(model.center.^2,1),[n 1])...
     ///       -2*x'*model.center)/(2*model.sigma^2));
    ///y=(K*model.alpha)';

    cout << "x.col(0): " << x.col(0) << endl;
    int d = x.rows(); //4
    int n = x.cols();//224 //1310720
    int b = model.center.cols(); //100 //132
    cout << "d=" << d << endl;
    cout << "n=" << n << endl;
    cout << "b=" << b << endl;

    VectorXd vec1(n), vec2(b), y(n);
    MatrixXd mat1(n,b), mat2(n,b), mat3(n,b);

    vec1 = (x.array()*x.array()).colwise().sum(); //different in 0.02
    //cout << "vec1 = " << vec1 << endl;
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
       //cout << xtrain_mean << endl;
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
        //cout << i << endl;
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
        int grid_step = 1;
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
        cout << "xnum = " << xnum << endl;
        cout << "ynum = " << ynum << endl;
        cout << "uvlength = " << uvlength << endl;
        //int cind = 512 + 640 * 1024;
        //cout << "cind = " << cind << endl;

        MatrixXi uv = MatrixXi::Zero(2, uvlength); //2 rows, unum cols
        MatrixXi uvdist = MatrixXi::Zero(2, uvlength); //2 rows, unum cols

        for(int j=0; j<ynum; j++){
            for(int i=0; i<xnum; i++){
                int ind = i + j*(xnum);
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
        ///MatrixXd xyz_gt(3, uvlength);
        MatrixXd uv_S2 = MatrixXd::Zero(2, uvlength);
        ///MatrixXd uvdouble(2,uvlength);
        MatrixXd X0test(3,uvlength), Y0test(3,uvlength);
        MatrixXd XS_W0_test(3,uvlength), XS_S_test(3,uvlength), aux(3,uvlength), UV_test(3,uvlength);
        MatrixXd xtest(4, uvlength), yte(4, uvlength);

        vec = -R_WE_test.transpose()*t_WE_test;
        //cout << vec << endl;
        //xyz_gt = R_WE_test.transpose()*(K_E_eig.lu().solve(matrix)) + vec.replicate(1, uvlength);
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


       /// uvdouble = uv.cast<double>();
        uv_S2.row(0) = (uv.cast<double>().row(0) - offsetw*VectorXd::Ones(uvlength).transpose()) / meter2pixel;
        uv_S2.row(1) = (uv.cast<double>().row(1) - offseth*VectorXd::Ones(uvlength).transpose()) / meter2pixel;
        //cout << uv_S2 << endl;

        X0test << uv_S2, MatrixXd::Zero(1, uvlength);
        X0test = R_SW_eig * X0test + t_SW_eig.replicate(1, uvlength);
        //cout << X0test << endl;
        /***************************************************************************************************************/


        intersectRayWithPlane(X0test, Y0test, t_SW_eig, R_SW_eig, XS_W0_test, XS_S_test);
        intersectRayWithPlane(X0test, Y0test, t_SW_z0_eig, R_SW_eig, aux, UV_test);

        xtest << UV_test.row(0), UV_test.row(1), XS_S_test.row(0), XS_S_test.row(1);
        //cout <<"xtest.col(0): \n" << xtest.col(0) <<  endl;
        //cout <<"xtest.col(545): \n" << xtest.col(545) << endl;

        kernel_regression_apply(xtest, KR_model_optical, yte);
        //cout << "yte.col(0): \n: "<< yte.col(0) << endl;
        //cout << "yte.col(545): \n: "<< yte.col(545) << endl;


        MatrixXd yte_m2p = yte * meter2pixel;
        for(int i = 0; i<uvlength; i++){
            uvdist(0,i) =  yte_m2p(2,i) + offsetw;
            uvdist(1,i) =  yte_m2p(3,i) + offseth;
        }



        int x_t = xnum/2;
        int y_t = ynum/2;
        int testind = x_t + y_t * xnum;
        cout << "testind = " << testind << endl;

        cout << " uv.col(testind): \n" << uv.col(testind) << "\n" <<  endl;
        //cout << " xtest.col(testind): \n" << xtest.col(testind) << "\n" <<  endl;
        //cout << " yte.col(testind): \n" << yte.col(testind) << "\n" <<  endl;
        cout << " uvdist.col(testind): \n" << uvdist.col(testind) << "\n" <<  endl;

        MatrixXi dist = MatrixXi::Zero(2, uvlength);
        dist.row(0) = uv.row(0) - uvdist.row(0);
        dist.row(1) = uv.row(1) - uvdist.row(1);

        cout << " dist.col(testind): \n" << dist.col(testind) << "\n" <<  endl;

        ///TODO: to try to cast uvdist to float
        MatrixXf uvdistf = MatrixXf::Zero(2, uvlength);
        uvdistf = uvdist.cast<float>();

        MatrixXf distf = MatrixXf::Zero(2, uvlength);
        distf = dist.cast<float>();



        ofstream myfile;

        myfile.open ("./maps/uv.txt");
        myfile << uv.transpose();
        myfile.close();

        myfile.open ("./maps/uvdistf.txt");
        myfile << uvdistf.transpose();
        myfile.close();

        myfile.open ("./maps/distf.txt");
        myfile << distf.transpose();
        myfile.close();

        /*
        typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_MxN;
        Matrix_MxN J = Matrix_MxN::Random(10,5);
        Eigen::write_binary("matrix.dat",J);
        std::cout << "\n original \n" << J << std::endl;
        Matrix_MxN J_copy;
        Eigen::read_binary("matrix.dat",J_copy);
        std::cout << "\n copy \n" << J_copy << std::endl;
        cout.flush();'
        */
        write_binary("./maps/uvdistbinaryf.dat", uvdistf);

}



int main(){
    /*****************************************************************************************************/
    //INPUT: KR_model_optical, R_EOW, t_EOW, offsetw, offseth, meter2pixel, K_E_eig
    //OUTPUT: mapping between uv and uvdist

    Matrix3d R_E0W_test = Matrix3d::Zero();
    Vector3d t_E0W_test = Vector3d::Zero();

    /*
    R_E0W_test <<    0.996226308432124,   0.031066683714960,  -0.081043220263370,
                -0.028404458041297,   0.999024955127270,   0.033798310554464,
                 0.082014200911078,  -0.031368777405414,   0.996137375392073;

    t_E0W_test <<  -0.029793673250077, -0.078804286612701, 0.118703726732110;
    */

    ///TODO: to read R_EW from file taken from HMD_calib_reverse_elena3.dfg

    R_E0W_test <<  0.99946964008749672, -0.017290305852553037, -0.027594997135257333,
                   -0.021733986259796501, 0.014517624200814155, 0.99511068624329713,
                  -0.097693196856066714, -0.09387251874592506, 0.02914922178950281;

    t_E0W_test << 0.097240770502497756, 0.99483393358949457, 0.10235504320132316;


    float offsetw = 0;
    float offseth = 0;


    Matrix3d K_E_eig = Matrix3d::Zero();
    K_E_eig << 1575,  0, -664.3,
               0, -1570.5, -509.6 ,
               0,   0,      -1;


    double meter2pixel;
    Vector3d t_SW_eig;
    Matrix3d R_SW_eig = Matrix3d::Zero();
    Vector3d t_SW_z0_eig = Vector3d::Zero();

    Vector3d t_WS_eig;
    Matrix3d R_WS_eig = Matrix3d::Zero();
    Vector3d t_WS_z0_eig = Vector3d::Zero();


    ///TODO: to read screen parameters R_SW, t_SW, meter2pixel from file

    /*
    meter2pixel = 1950.9;

    t_SW_eig(0) =  -0.3696; t_SW_eig(1) = -0.3621; t_SW_eig(2) =  -0.6137;

    R_SW_eig << 0.9996,   -0.0237,    0.0150,
                0.0240,    0.9996,   -0.0179,
               -0.0145,    0.0183,    0.9997;
     */


    meter2pixel = 1656.5;

    t_WS_eig(0) =  0.0485; t_WS_eig(1) = 0.1055; t_WS_eig(2) =  0.7423;

    R_WS_eig << 0.9997,   0.0242,    -0.0083,
               -0.0240,    0.9996,   0.0169,
                0.0087,    -0.0167,    0.9998;

    R_SW_eig = R_WS_eig.transpose();
    t_SW_eig = - R_WS_eig.transpose() * t_WS_eig;





    t_SW_z0_eig = t_SW_eig;
    t_SW_z0_eig(2) = 0;


    //read KR_model from set of files in the folder
    KR_model_vs KR_model_optical_fromFile;
    read_KR_model_vs_fromFiles(&KR_model_optical_fromFile);
    //print_KR_model_vs(&KR_model_optical_fromFile);

    computeDistortionMap(KR_model_optical_fromFile, R_E0W_test, t_E0W_test,  offsetw,  offseth,  meter2pixel, R_SW_eig, t_SW_eig, t_SW_z0_eig, K_E_eig);




}
