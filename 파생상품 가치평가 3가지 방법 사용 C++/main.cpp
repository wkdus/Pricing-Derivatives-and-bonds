#include<iostream>
#include <iomanip>
#include"option.h"

void calcPrice(Option& option, PricingMethod pm);
void print(Option& p);
int main() {
	//클래스 객체 생성

	PlainVanillaOption a(0.1, 250, Call);
	PlainVanillaOption b(0.2, 260, Put);
	DigitalOption c(0.15, 240, Put);
	PlainVanillaOption d(0.25, 270, Call);
	DigitalOption e(0.2, 230, Call);

	std::cout << "-------------------------------------------------" << std::endl;
	std::cout <<std::setw(3)<<"No"
		<<std::setw(15)<<"Analytic"
		<<std::setw(15)<<"MonteCarlo"
		<<std::setw(15)<<"Binomial" << std::endl;
	std::cout << "-------------------------------------------------" << std::endl;
	print(a);
	print(b);
	print(c);
	print(d);
	print(e);

}
void calcPrice(Option& option, PricingMethod pm) {
	option.setPricingMethod(pm);
	std::cout<<std::setw(15)<< option.npv();
}
void print(Option& p) {
	static int n = 1;
	std::cout <<std::setw(3)<< n;
	n++;
		calcPrice(p, Analytic);
		calcPrice(p, MonteCarlo);
		calcPrice(p, BinomialTree);
		std::cout << std::endl;
		std::cout << "-------------------------------------------------" << std::endl;
}