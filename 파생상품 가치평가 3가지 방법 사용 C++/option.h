#pragma once
#include"pricing_methods.h"
#include<vector>

class MarketVariables {
/*변경한 부분*/
private:
	double stock_ ;
	double interestRate_ ;
	double dividend_ ;
	double volatility_ ;
public:
	MarketVariables(double s=250, double i=0.02, double d=0.01, double v=0.2)
		: stock_(s), interestRate_(i), dividend_(d), volatility_(v) {}

	~MarketVariables(){}
	std::vector<double> getMarketVariables() const;
	void setMarketVariables(double s, double i, double d, double v);
};

class Option {
protected:
	double maturity_;
	double exercise_;
	OptionType optiontype_;
	PricingMethod pricingMethod_ = Analytic;
	MarketVariables marketVariables_;
	std::vector<double>M = marketVariables_.getMarketVariables();
	Pricing pricing = { M[0], M[1], M[2], M[3], exercise_, maturity_, optiontype_ };
public:
	Option(double m, double e, OptionType ot);
	~Option() {}
	virtual double npv() = 0;
	void setPricingMethod(PricingMethod pm) { pricingMethod_ = pm; }
	void setMarketVariables(MarketVariables mv);
};

class PlainVanillaOption : public Option {
public:
	PlainVanillaOption(double m, double e, OptionType ot)
		: Option(m, e, ot) {}
	~PlainVanillaOption() {}
	virtual double npv();
};
class DigitalOption :public Option {
public:
	DigitalOption(double m, double e, OptionType ot)
		: Option(m, e, ot) {}
	~DigitalOption() {}
	virtual double npv();
};