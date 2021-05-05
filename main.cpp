#include <iostream>
#include <vector>
#include <fstream>
#include <random>
using namespace std;
double myrandom(){
    static default_random_engine e(clock());
    uniform_real_distribution<double> u(-1.0,1.0);
    return u(e);
}
class mydata{
public:
    void read(){
        ifstream infile,outfile;
        char ch;
        double temp;
        infile.open("in.txt",ios::in);
        outfile.open("out.txt", ios::in);
        if(!infile||!outfile){
            fprintf(stderr,"open file error");
            exit(1);
        }
        int j=0;
        while(infile>>temp){

            d_in.emplace_back();
            d_in[j].push_back(temp);
            for(int i=0;i<In-1;i++){
                infile>>ch>>temp;
                d_in[j].push_back(temp);
            }
            j++;

        }
        infile.close();
        j=0;
        while(outfile >> temp){
            d_out.emplace_back();
            d_out[j].push_back(temp);
            for(int i=0;i<Out-1;i++){
                outfile>>ch>>temp;
                d_out[j].push_back(temp);
            }
            j++;
        }
        outfile.close();
    }
    void init(){
        for(int i=0;i<In;i++){
            Minin[i]=Maxin[i]=d_in[0][i];
            for(auto & j : d_in){
                Maxin[i]=Maxin[i]>j[i]?Maxin[i]:j[i];
                Minin[i]=Minin[i]<j[i]?Minin[i]:j[i];
            }
        }
        for(int i=0;i<Out;i++){
            Minout[i]=Maxout[i]=d_out[0][i];
            for(auto & j : d_out){
                Maxout[i]=Maxout[i]>j[i]?Maxout[i]:j[i];
                Minout[i]=Minout[i]<j[i]?Minout[i]:j[i];
            }
        }
        for(int i=0;i<In;i++){
            for(auto & j : d_in){
                j[i]=(j[i]-Minin[i])/(Maxin[i]-Minin[i]);
            }
        }
        for(int i=0;i<Out;i++){
            for(auto & j : d_out){
                j[i]=(j[i]-Minout[i])/(Maxout[i]-Minout[i]);
            }
        }
        for(auto & i : v){
            for(double & j : i){
                j=myrandom();
            }
        }
        for(auto & i : w){
            for(double & j : i){
                j=myrandom();
            }
        }
        for(auto & i : u){
            for(double & j : i){
                j=myrandom();
            }
        }
    }
    void train(){
        int count=1;
        do{
            mse=0;
            for(int i=0;i<d_in.size();i++){
                computo(i);
                backupdate(i);
                for(int j=0;j<Out;j++){
                    double tmp1=Outputdata[j]*(Maxout[0]-Minout[0])+Minout[0];
                    double tmp2=d_out[i][j]*(Maxout[0]-Minout[0])+Minout[0];
                    mse+=(tmp1-tmp2)*(tmp1-tmp2);
                }
            }mse=mse/(double)d_in.size()*Out;
            if(count%1000==0){
                cout<<count<<"   "<<mse<<endl;
            }
            count++;
        }while(mse>=4);
        cout<<"train finished"<<endl;
    }
    void computo(int var){
        double sum;
        for(int i=0;i<Neuron;i++){
            sum=0;
            for(int j=0;j<In;j++){
                sum+=v[i][j]*d_in[var][j];
            }
            y[i]=1/(1+exp(-1*sum));
        }
        for(int i=0;i<Neuron;i++){
            sum=0;
            for(int j=0;j<Neuron;j++){
                sum+=u[i][j]*y[j];
            }
            z[i]=1/(1+exp(-1*sum));
        }
        for(int i=0;i<Out;i++){
            sum=0;
            for(int j=0;j<Neuron;j++){
                sum+=w[i][j]*z[j];
            }
            Outputdata[i]=1/(1+exp(-1*sum));
        }
    }
    void backupdate(int var){
        double t,t2;
        for(int i=0;i<In;i++){
            t2 = 0;
            for (int k = 0; k < Neuron; k++) {
                t = 0;
                for (int l = 0; l < Out; l++) {
                    dw[l][k] = Walta * (d_out[var][l] - Outputdata[l]) * Outputdata[l] * (1 - Outputdata[l]) * y[k];
                    w[l][k] += dw[l][k];
                    t += (d_out[var][l] - Outputdata[l]) * Outputdata[l] * (1 - Outputdata[l]) * w[l][k];
                }
                for (int j = 0; j < Neuron; j++) {
                    du[k][j] = Ualta * t * z[k] * (1 - z[k]) * y[j];
                    u[k][j]+=du[k][j];
                    t2 += Ualta * t * z[k] * (1 - z[k]) * u[k][j];
                }
            }
            for(int j=0;j<Neuron;j++){
                dv[j][i]=Valta*t2*y[j]*(1-y[j])*d_in[var][i];
                v[j][i]+=dv[j][i];
            }
        }
    }
    void test(){
        ifstream infile;
        double temp,sum;
        char ch;

        infile.open("test.txt",ios::in);
        if(!infile){
            fprintf(stderr,"open file error");
            exit(0);
        }
        int j=0;
        while(infile>>temp){
            t_in.emplace_back();
            t_out.emplace_back();
            t_in[j].push_back(temp);
            for(int i=0;i<In+Out-1;i++){
                infile>>ch>>temp;
                if(i<In-1){
                    t_in[j].push_back(temp);
                }
                else t_out[j].push_back(temp);
            }
            j++;
        }
        infile.close();
        pre.resize(t_in.size());
        for(int i=0;i<In;i++){
            for(j=0;j<t_in.size();j++){
                t_in[j][i]=(t_in[j][i]-Minin[i])/(Maxin[i]-Minin[i]);
            }
        }

        for(int k=0;k<t_in.size();k++){
            for(int i=0;i<Neuron;i++){
                sum=0;
                for(j=0;j<In;j++){
                    sum+=v[i][j]*t_in[k][j];
                }
                y[i]=1/(1+exp(-1*sum));
            }
            for(int i=0;i<Neuron;i++){
                sum=0;
                for(j=0;j<Neuron;j++){
                    sum+=u[i][j]*y[j];
                }
                z[i]=1/(1+exp(-1*sum));
            }
            sum=0;
            for(j=0;j<Neuron;j++){
                sum+=w[0][j]*z[j];
            }
            pre[k].push_back(1/(1+exp(-1*sum))*(Maxout[0]-Minout[0])+Minout[0]);
            cout<<"number:"<<k+1<<"  predict:"<<pre[k][0]<<"  fact:"<<t_out[k][0]<<endl;
        }
        for(int k=0;k<t_in.size();k++){
            for(int l=0;l<Out;l++){
                rmse+=(pre[k][l]-t_out[k][l])*(pre[k][l]-t_out[k][l]);
            }
        }
        rmse= sqrt(rmse/(double)t_in.size());
        cout<<"rmse:"<<rmse<<endl;
    }
private:
    static const int In=13,Out=1,Neuron=40,Trainc=5000;
    constexpr static const double Walta=0.05,Valta=0.05,Ualta=0.01;
    vector<vector<double> > d_in,d_out,t_in,t_out,pre;
    double v[Neuron][In]{},y[Neuron]{},z[Neuron]{},w[Out][Neuron]{},dv[Neuron][In]{},dw[Out][Neuron]{},u[Neuron][Neuron]{},du[Neuron][Neuron]{};
    double Maxin[In]{},Minin[In]{},Maxout[Out]{},Minout[Out]{},Outputdata[Out]{};
    double mse{},rmse{};
};
int main() {
    mydata d;
    d.read();
    d.init();
    d.train();
    d.test();
    return 0;
}
