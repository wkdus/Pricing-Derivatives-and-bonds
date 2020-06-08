#include"option.h"

std::vector<double> MarketVariables::getMarketVariables() const {
	std::vector<double> MV;
	MV.push_back(stock_);
	MV.push_back(interestRate_);
	MV.push_back(dividend_);
	MV.push_back(volatility_);
	return MV;
}
void MarketVariables::setMarketVariables(double s, double i, double d, double v) {
	stock_ = s;		interestRate_ = i;		dividend_ = d;		volatility_ = v;
}
Option::Option(double m, double e, OptionType ot)
	: maturity_(m), exercise_(e), optiontype_(ot) {}
void Option::setMarketVariables(MarketVariables mv) {
	std::vector<double>m = mv.getMarketVariables();
	marketVariables_.setMarketVariables(m[0], m[1], m[2], m[3]);
}
double PlainVanillaOption::npv() {
	if (pricingMethod_ == 0)
		return pricing.bsprice(PlainVanilla);
	else if (pricingMethod_ == 1)
		return pricing.mcprice(PlainVanilla, numOfSimulation);
	else
		return pricing.bntprice(PlainVanilla, nsteps);
}
double DigitalOption::npv() {
	if (pricingMethod_ == 0)
		return pricing.bsprice(Digital);
	else if (pricingMethod_ == 1)
		return pricing.mcprice(Digital, numOfSimulation);
	else
		return pricing.bntprice(Digital, nsteps);
}