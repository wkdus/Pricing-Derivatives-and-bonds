#pragma once

/*기본정보*/
enum OptionType { Call = 1, Put = -1 };
enum PricingMethod { Analytic, MonteCarlo, BinomialTree };
enum PriceType { PlainVanilla, Digital };

const unsigned int numOfSimulation = 100000;
const unsigned int nsteps = 300;

double payoff(PriceType p, double s, double e, OptionType t);
double normpdf(double x, double mu = 0, double sigma = 1);
double normcdf(double x, double mu = 0, double sigma = 1);

/*PricingTool*/
class Pricing {
private:
	double s;
	double r;
	double q;
	double sigma;
	double k;
	double t;
	OptionType type;
public:
	Pricing(double s, double r, double q, double sigma, double k, double t, OptionType optiontype)
		: s(s), r(r), q(q), sigma(sigma), k(k), t(t), type(optiontype) {}
	~Pricing() {}
	double bsprice(PriceType pricetype);
	double mcprice(PriceType pricetype, unsigned int numOfSimulation);
	double bntprice(PriceType pricetype, unsigned int nsteps);
};