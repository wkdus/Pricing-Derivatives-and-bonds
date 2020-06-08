#include"pricing_methods.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <random>

double payoff(PriceType p, double s, double e, OptionType t) {
	if (p == 0) {
		return (t*(s - e)>0) ? t*(s - e) : 0;
	}
	else
		return (t*(s - e) > 0) ? 1 : 0;
}
double normpdf(double x, double mu, double sigma)
{
	return 1 / sqrt(2 * M_PI) / sigma*exp(-0.5*pow((x - mu) / sigma, 2));
}
double normcdf(double x, double mu, double sigma)
{
	double v = (x - mu) / sigma;
	return 0.5 * erfc(-v * M_SQRT1_2);
}

/*Analytic*/
double Pricing::bsprice(PriceType pricetype) {
	double d1 = (log(s / k) + (r - q + 0.5*sigma*sigma)*t) / (sigma*sqrt(t));
	double d2 = d1 - sigma*sqrt(t);
	double nd1 = normcdf(type*d1);
	double nd2 = normcdf(type*d2);
	double price;
	if (pricetype == 0)
		price = type*(s*exp(-q*t)*nd1 - k*exp(-r*t)*nd2);
	else if (pricetype == 1)
		price = exp(-r*t)*nd2;
	return price;
}

/*Binomial Tree*/
double Pricing::bntprice(PriceType pricetype, unsigned int nsteps) {

	double dt = t / nsteps;
	double u = exp(sigma*sqrt(dt));
	double d = 1 / u;
	double p = (exp((r - q)*dt) - d) / (u - d);
	double df = exp(-r*dt);

	std::vector<double> v(nsteps + 1, 0.0);
	for (int j = 0; j <= nsteps; ++j) {
		double st = s*pow(u, nsteps - j)*pow(d, j);
		v[j] = payoff(pricetype,st,k,type);
	}

	for (int i = nsteps - 1; i >= 0; --i) {
		for (int j = 0; j <= i; ++j) {
			v[j] = df*(v[j] * p + v[j + 1] * (1 - p));
		}
	}
	return v[0];
}

/*MonteCarlo*/
double Pricing::mcprice(PriceType pricetype, unsigned int numOfSimulation) {

	double sumOfPayoff = 0;
	double df = exp(-r*t);

	std::mt19937_64 gen;
	std::normal_distribution<double> engine(0.0, 1.0);
	gen.seed(std::random_device{}());
	double es = s*exp((r - q - 0.5*sigma*sigma)*t);
	double diffution = sigma*sqrt(t);
	for (unsigned int i = 0; i < numOfSimulation; ++i) {
		double e = engine(gen);
		for (int j = 0; j < 2; ++j) {
			double st = es * exp(diffution*(1 - j * 2)*e);  //antithetic method
			double p = payoff(pricetype,st,k,type);
			sumOfPayoff += df * p;
		}
	}
	return sumOfPayoff / numOfSimulation / 2.0;
}
