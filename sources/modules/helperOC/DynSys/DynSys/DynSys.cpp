#include <limits>
#include <array>
#include <typeinfo>
#if defined(WIN32)	// Windows
#pragma warning(disable : 4996)
#endif	/* Windows */
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/stepper/adams_bashforth_moulton.hpp>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include "DynSys_impl.hpp"
#include <levelset/Grids/HJI_Grid.hpp>
//#include <opencv2/opencv.hpp>
#include <sstream>
#include <Core/UVec.hpp>

bool HPxPy::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr) {
	bool result = true;
	if (!XData.empty())result &= save_vector(XData, std::string("XData"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!YData.empty())result &= save_vector(YData, std::string("YData"), beacls::IntegerVec(), true, fs, variable_ptr);
	result &= save_value(UData, std::string("UData"), true, fs, variable_ptr);
	result &= save_value(VData, std::string("VData"), true, fs, variable_ptr);

	if (!Color.empty())result &= save_vector(Color, std::string("Color"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!MarkerFaceColor.empty())result &= save_vector(MarkerFaceColor, std::string("MarkerFaceColor"), beacls::IntegerVec(), true, fs, variable_ptr);
	result &= save_value(MarkerSize, std::string("MarkerSize"), true, fs, variable_ptr);
	result &= save_value(MaxHeadSize, std::string("MaxHeadSize"), true, fs, variable_ptr);
	result &= save_value(LineWidth, std::string("LineWidth"), true, fs, variable_ptr);
	return result;
}
HPxPy::HPxPy(
	beacls::MatFStream* fs,
	beacls::MatVariable* var) {
	beacls::IntegerVec dummy;
	load_vector(XData, std::string("XData"), dummy, true, fs, var);
	load_vector(YData, std::string("YData"), dummy, true, fs, var);
	load_value(UData, std::string("UData"), true, fs, var);
	load_value(VData, std::string("VData"), true, fs, var);

	load_vector(Color, std::string("Color"), dummy, true, fs, var);
	load_vector(MarkerFaceColor, std::string("MarkerFaceColor"), dummy, true, fs, var);
	load_value(MarkerSize, std::string("MarkerSize"), true, fs, var);
	load_value(MaxHeadSize, std::string("MaxHeadSize"), true, fs, var);
	load_value(LineWidth, std::string("LineWidth"), true, fs, var);
}
HPxPy::HPxPy() :
	XData(0),
	YData(0),
	UData(0),
	VData(0),
	Marker(std::string()),
	Color(beacls::FloatVec()),
	MarkerFaceColor(beacls::FloatVec()),
	MarkerSize(0),
	MaxHeadSize(0),
	LineWidth(0)

{
}
bool HPxPy::operator==(const HPxPy& rhs) const {
	if (this == &rhs) return true;
	else if (UData != rhs.UData) return false;
	else if (VData != rhs.VData) return false;
	else if (Marker != rhs.Marker) return false;
	else if (MarkerSize != rhs.MarkerSize) return false;
	else if (MaxHeadSize != rhs.MaxHeadSize) return false;
	else if (LineWidth != rhs.LineWidth) return false;
	if ((XData.size() != rhs.XData.size()) || !std::equal(XData.cbegin(), XData.cend(), rhs.XData.cbegin())) return false;
	if ((YData.size() != rhs.YData.size()) || !std::equal(YData.cbegin(), YData.cend(), rhs.YData.cbegin())) return false;
	if ((Color.size() != rhs.Color.size()) || !std::equal(Color.cbegin(), Color.cend(), rhs.Color.cbegin())) return false;
	if ((MarkerFaceColor.size() != rhs.MarkerFaceColor.size()) || !std::equal(MarkerFaceColor.cbegin(), MarkerFaceColor.cend(), rhs.MarkerFaceColor.cbegin())) return false;

	return true;
}


DynSys_impl::DynSys_impl() :
	nx(0),
	nu(0),
	nd(0),
	x(beacls::FloatVec()),
	u(std::vector<beacls::FloatVec>()),
	xhist(std::vector<beacls::FloatVec>()),
	uhist(std::vector<std::vector<beacls::FloatVec>>()),
	pdim(beacls::IntegerVec()),
	vdim(beacls::IntegerVec()),
	hdim(beacls::IntegerVec()),
	TIdims(beacls::IntegerVec()),
	hpxpy(HPxPy()),
	hpxpyhist(std::vector<HPxPy>()),
	hvxvy(beacls::FloatVec()),
	hvxvyhist(std::vector<beacls::FloatVec>()),
	hpv(beacls::FloatVec{1,1}),
	hpvhist(std::vector<beacls::FloatVec>{beacls::FloatVec{1, 1}}),
	data(beacls::FloatVec()),
	partialFunction(NULL)
{
}
DynSys_impl::DynSys_impl(
	const size_t nx,
	const size_t nu,
	const size_t nd,
	const beacls::IntegerVec& pdim,
	const beacls::IntegerVec& hdim,
	const beacls::IntegerVec& vdim,
	const beacls::IntegerVec& TIdims) : 
	nx(nx),
	nu(nu),
	nd(nd),
	x(beacls::FloatVec()),
	u(std::vector<beacls::FloatVec>()),
	xhist(std::vector<beacls::FloatVec>()),
	uhist(std::vector<std::vector<beacls::FloatVec>>()),
	pdim(pdim),
	vdim(vdim),
	hdim(hdim),
	TIdims(TIdims),
	hpxpy(HPxPy()),
	hpxpyhist(std::vector<HPxPy>()),
	hvxvy(beacls::FloatVec()),
	hvxvyhist(std::vector<beacls::FloatVec>()),
	hpv(beacls::FloatVec{1, 1}),
	hpvhist(std::vector<beacls::FloatVec>{beacls::FloatVec{1, 1}}),
	data(beacls::FloatVec()),
	partialFunction(NULL)
{
}
DynSys_impl::DynSys_impl(
	const size_t nx,
	const size_t nu,
	const size_t nd,
	const beacls::IntegerVec& pdim,
	const beacls::IntegerVec& hdim,
	const beacls::IntegerVec& vdim,
	const beacls::IntegerVec& TIdims,
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	nx(nx),
	nu(nu),
	nd(nd),
	x(beacls::FloatVec()),
	u(std::vector<beacls::FloatVec>()),
	xhist(std::vector<beacls::FloatVec>()),
	uhist(std::vector<std::vector<beacls::FloatVec>>()),
	pdim(pdim),
	vdim(vdim),
	hdim(hdim),
	TIdims(TIdims),
	hpxpy(HPxPy()),
	hpxpyhist(std::vector<HPxPy>()),
	hvxvy(beacls::FloatVec()),
	hvxvyhist(std::vector<beacls::FloatVec>()),
	hpv(beacls::FloatVec{1, 1}),
	hpvhist(std::vector<beacls::FloatVec>{beacls::FloatVec{1, 1}}),
	data(beacls::FloatVec()),
	partialFunction(NULL) {

	beacls::IntegerVec dummy;
	load_vector(x, std::string("x"), dummy, true, fs, variable_ptr);
	load_vector_of_vectors(u, std::string("u"), dummy, true, fs, variable_ptr);

	load_vector_of_vectors(xhist, std::string("xhist"), dummy, true, fs, variable_ptr);

	beacls::MatVariable* uhist_variable_ptr = beacls::getVariableFromStruct(variable_ptr, std::string("uhist"));
	uhist.resize(beacls::getCellSize(uhist_variable_ptr));
	for (size_t i = 0; i < uhist.size(); ++i) {
		load_vector_of_vectors(uhist[i], std::string(), dummy, true, fs, uhist_variable_ptr, i);
	}
	beacls::closeMatVariable(uhist_variable_ptr);



	beacls::MatVariable* hpxpy_var = beacls::getVariableFromStruct(variable_ptr, std::string("hpxpy"));
	hpxpy = HPxPy(fs, hpxpy_var);
	beacls::closeMatVariable(hpxpy_var);

	beacls::MatVariable* hpxpyhist_var = beacls::getVariableFromStruct(variable_ptr, std::string("hpxpyhist"));
	hpxpyhist.resize(beacls::getCellSize(hpxpyhist_var));
	for (size_t i = 0; i < hpxpyhist.size(); ++i) {
		beacls::MatVariable* hpxpyhist_i_var = beacls::getVariableFromCell(hpxpyhist_var, i);
		hpxpyhist[i] = HPxPy(fs, hpxpyhist_i_var);
		beacls::closeMatVariable(hpxpyhist_i_var);

	}
	beacls::closeMatVariable(hpxpyhist_var);

	load_vector(hvxvy, std::string("hvxvy"), dummy, true, fs, variable_ptr);
	load_vector_of_vectors(hvxvyhist, std::string("hvxvyhist"), dummy, true, fs, variable_ptr);

	load_vector(hpv, std::string("hpv"), dummy, true, fs, variable_ptr);
	load_vector_of_vectors(hpvhist, std::string("hpvhist"), dummy, true, fs, variable_ptr);

	load_vector(data, std::string("data"), dummy, true, fs, variable_ptr);
}
DynSys_impl::~DynSys_impl() {
}
bool DynSys_impl::operator==(const DynSys_impl& rhs) const {
	if (this == &rhs) return true;
	else if (nx != rhs.nx) return false;
	else if (nu != rhs.nu) return false;
	else if (nd != rhs.nd) return false;
	else if (!hpxpy.operator==(rhs.hpxpy)) return false;
	else if (partialFunction != rhs.partialFunction) return false;
	else if ((x.size() != rhs.x.size()) || !std::equal(x.cbegin(), x.cend(), rhs.x.cbegin())) return false;
	else if ((pdim.size() != rhs.pdim.size()) || !std::equal(pdim.cbegin(), pdim.cend(), rhs.pdim.cbegin())) return false;
	else if ((vdim.size() != rhs.vdim.size()) || !std::equal(vdim.cbegin(), vdim.cend(), rhs.vdim.cbegin())) return false;
	else if ((hdim.size() != rhs.hdim.size()) || !std::equal(hdim.cbegin(), hdim.cend(), rhs.hdim.cbegin())) return false;
	else if ((TIdims.size() != rhs.TIdims.size()) || !std::equal(TIdims.cbegin(), TIdims.cend(), rhs.TIdims.cbegin())) return false;
	else if ((hpxpyhist.size() != rhs.hpxpyhist.size()) || !std::equal(hpxpyhist.cbegin(), hpxpyhist.cend(), rhs.hpxpyhist.cbegin())) return false;
	else if ((hvxvy.size() != rhs.hvxvy.size()) || !std::equal(hvxvy.cbegin(), hvxvy.cend(), rhs.hvxvy.cbegin())) return false;
	else if ((hpv.size() != rhs.hpv.size()) || !std::equal(hpv.cbegin(), hpv.cend(), rhs.hpv.cbegin())) return false;
	else if ((data.size() != rhs.data.size()) || !std::equal(data.cbegin(), data.cend(), rhs.data.cbegin())) return false;
	else if ((u.size() != rhs.u.size()) || !std::equal(u.cbegin(), u.cend(), rhs.u.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;
	else if ((xhist.size() != rhs.xhist.size()) || !std::equal(xhist.cbegin(), xhist.cend(), rhs.xhist.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;
	else if ((hvxvyhist.size() != rhs.hvxvyhist.size()) || !std::equal(hvxvyhist.cbegin(), hvxvyhist.cend(), rhs.hvxvyhist.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;
	else if ((hpvhist.size() != rhs.hpvhist.size()) || !std::equal(hpvhist.cbegin(), hpvhist.cend(), rhs.hpvhist.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;
	else if ((uhist.size() != rhs.uhist.size()) || !std::equal(uhist.cbegin(), uhist.cend(), rhs.uhist.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin(), [](const auto& lhs, const auto& rhs) {
			return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
		}));
	})) return false;
	else return true;
}
DynSys::DynSys() {
	pimpl = new DynSys_impl;

}

DynSys::DynSys(
	const size_t nx,
	const size_t nu,
	const size_t nd,
	const beacls::IntegerVec& pdim,
	const beacls::IntegerVec& hdim,
	const beacls::IntegerVec& vdim,
	const beacls::IntegerVec& TIdims)
{
	pimpl = new DynSys_impl(nx, nu, nd, pdim, hdim, vdim, TIdims);
}
DynSys::DynSys(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	size_t nx;	//!< Number of state dimensions
	size_t nu;	//!< Number of control inputs
	size_t nd;	//!< Number of disturbance dimensions
	load_value(nx, std::string("nx"), true, fs, variable_ptr);
	load_value(nu, std::string("nu"), true, fs, variable_ptr);
	load_value(nd, std::string("nd"), true, fs, variable_ptr);
	beacls::IntegerVec dummy;
	beacls::IntegerVec pdim;	//!< position dimensions
	beacls::IntegerVec vdim;	//!< velocity dimensions
	beacls::IntegerVec hdim;	//!< heading dimensions
	beacls::IntegerVec TIdims;	//!< TI dimensions
	load_vector(pdim, std::string("pdim"), dummy, true, fs, variable_ptr);
	load_vector(vdim, std::string("vdim"), dummy, true, fs, variable_ptr);
	load_vector(hdim, std::string("hdim"), dummy, true, fs, variable_ptr);
	load_vector(TIdims, std::string("TIdims"), dummy, true, fs, variable_ptr);
	pimpl = new DynSys_impl(nx, nu, nd, pdim, hdim, vdim, TIdims, fs, variable_ptr);
}
DynSys::~DynSys() {
	if (pimpl) delete pimpl;
}
bool DynSys::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	if (!pimpl) {
		if (!rhs.pimpl) return true;
		else return false;
	}
	else {
		if (!rhs.pimpl) return false;
		else {
			if (pimpl == rhs.pimpl) return true;
			else if (*pimpl == *rhs.pimpl) return true;
			else return false;
		}
	}
}

bool DynSys_impl::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = save_value(nx, std::string("nx"), true, fs, variable_ptr);
	result &= save_value(nu, std::string("nu"), true, fs, variable_ptr);
	result &= save_value(nd, std::string("nd"), true, fs, variable_ptr);

	if (!x.empty()) result &= save_vector(x, std::string("x"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!u.empty())result &= save_vector_of_vectors(u, std::string("u"), beacls::IntegerVec(), true, fs, variable_ptr);

	if (!xhist.empty())result &= save_vector_of_vectors(xhist, std::string("xhist"), beacls::IntegerVec(), true, fs, variable_ptr);

	beacls::MatVariable* uhist_variable_ptr = beacls::createMatCell(std::string("uhist"), uhist.size());
	for (size_t i = 0; i < uhist.size(); ++i) {
		if (!uhist[i].empty()) result &= save_vector_of_vectors(uhist[i], std::string(), beacls::IntegerVec(), true, fs, uhist_variable_ptr, i);
	}
	beacls::setVariableToStruct(variable_ptr, uhist_variable_ptr, std::string("uhist"));
	beacls::closeMatVariable(uhist_variable_ptr);

	if (!pdim.empty())result &= save_vector(pdim, std::string("pdim"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!vdim.empty())result &= save_vector(vdim, std::string("vdim"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!hdim.empty())result &= save_vector(hdim, std::string("hdim"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!TIdims.empty())result &= save_vector(TIdims, std::string("TIdims"), beacls::IntegerVec(), true, fs, variable_ptr);

	beacls::MatVariable* hpxpy_var = beacls::createMatStruct(std::string("hpxpy"));
	result &= hpxpy.save(fs, hpxpy_var);
	beacls::setVariableToStruct(variable_ptr, hpxpy_var, std::string("hpxpy"));
	beacls::closeMatVariable(hpxpy_var);

	beacls::MatVariable* hpxpyhist_var = beacls::createMatCell(std::string("hpxpyhist"), hpxpyhist.size());
	for (size_t i = 0; i < hpxpyhist.size(); ++i) {
		beacls::MatVariable* hpxpyhist_i_var = beacls::createMatStruct(std::string("hpxpy"));
		result &= hpxpyhist[i].save(fs, hpxpyhist_i_var);
		beacls::setVariableToCell(hpxpyhist_var, hpxpyhist_i_var, i);
		beacls::closeMatVariable(hpxpyhist_i_var);
	}
	beacls::setVariableToStruct(variable_ptr, hpxpyhist_var, std::string("hpxpyhist"));
	beacls::closeMatVariable(hpxpyhist_var);

	if (!hvxvy.empty())result &= save_vector(hvxvy, std::string("hvxvy"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!hvxvyhist.empty())result &= save_vector_of_vectors(hvxvyhist, std::string("hvxvyhist"), beacls::IntegerVec(), true, fs, variable_ptr);

	if (!hpv.empty())result &= save_vector(hpv, std::string("hpv"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!hpvhist.empty())result &= save_vector_of_vectors(hpvhist, std::string("hpvhist"), beacls::IntegerVec(), true, fs, variable_ptr);

	if (!data.empty())result &= save_vector(data, std::string("data"), beacls::IntegerVec(), true, fs, variable_ptr);
	return result;
}

bool DynSys::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	if (pimpl) return pimpl->save(fs, variable_ptr);
	return false;
}


typedef std::vector< FLOAT_TYPE > OdeDataType;

class OdeState {
private:
	const DynSys* dynSys;
	const std::vector<beacls::FloatVec >& src_u;
	const std::vector<beacls::FloatVec >& src_d;
	beacls::IntegerVec u_sizes;
	beacls::IntegerVec d_sizes;
public:

	OdeState(
		const DynSys* dynSys,
		const std::vector<beacls::FloatVec >& src_u,
		const std::vector<beacls::FloatVec >& src_d
	) : dynSys(dynSys),
		src_u(src_u), src_d(src_d) {
		u_sizes.resize(src_u.size());
		std::transform(src_u.cbegin(), src_u.cend(), u_sizes.begin(), [](const auto& rhs) { return rhs.size(); });
		d_sizes.resize(src_d.size());
		std::transform(src_d.cbegin(), src_d.cend(), d_sizes.begin(), [](const auto& rhs) { return rhs.size(); });
	}

	void operator()(const OdeDataType& x, OdeDataType& dxdt, const FLOAT_TYPE t) {
		std::vector<beacls::FloatVec::const_iterator> x_ites(x.size());
		beacls::IntegerVec x_sizes(x.size());
		for (size_t i = 0; i < x.size(); i++) {
			x_ites[i] = x.cbegin() + i;
			x_sizes[i] = 1;
		}

		const size_t nx = dynSys->get_nx();
		std::vector<beacls::FloatVec> dx(nx);
		dynSys->dynamics(
			dx, t, x_ites, src_u, src_d, x_sizes
		);
		dxdt.resize(nx);
		std::transform(dx.cbegin(), dx.cend(), dxdt.begin(), [](const auto& rhs) { return rhs[0]; });
	}
};

bool DynSys_impl::ode113(
	beacls::FloatVec& dst_x,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& src_u,
	const std::vector<beacls::FloatVec >& src_d,
	const beacls::FloatVec& tspan,
	const beacls::FloatVec& x0
) {
	OdeState odeState(dynSys, src_u, src_d);
	size_t step = 10;
	const FLOAT_TYPE dt = (tspan[1] - tspan[0]) / step;
	beacls::FloatVec xtmp = x0;
	//	std::pair<FLOAT_TYPE, FLOAT_TYPE> range{ tspan[0], tspan[1] };
	boost::sub_range<const beacls::FloatVec> tspan_range(tspan);
	boost::numeric::odeint::adams_bashforth_moulton<8, OdeDataType> stepper;
	boost::numeric::odeint::integrate_adaptive(stepper, odeState, xtmp, tspan[0], tspan[1], dt,
		[&dst_x](const OdeDataType& x, const FLOAT_TYPE) {
		dst_x = x;
	});
	return true;
}
bool DynSys::getHeading(beacls::FloatVec& h, std::vector<beacls::FloatVec>& hhist) const {
	if (!pimpl) return false;
	beacls::IntegerVec hdim = pimpl->get_hdim();
	beacls::FloatVec x = pimpl->get_x();
	std::vector<beacls::FloatVec> xhist = pimpl->get_xhist();

	//!< If heading is a state
	if (!hdim.empty()) {
		h.resize(hdim.size());
		std::transform(hdim.cbegin(), hdim.cend(), h.begin(), [&x](const auto& rhs) { return x[rhs]; });
		hhist.resize(xhist.size());
		beacls::IntegerVec pdim = pimpl->get_pdim();
		std::transform(xhist.cbegin(), xhist.cend(), hhist.begin(), [&hdim, &pdim](const auto& xhist_i) {
			beacls::FloatVec x(pdim.size());
			std::transform(pdim.cbegin(), pdim.cend(), x.begin(), [&xhist_i](const auto& rhs) { return xhist_i[rhs]; });
			return x;
		});
		return true;
	}

	beacls::IntegerVec vdim = pimpl->get_vdim();
	//!< If vehicle has at least 2 velocity dimensions
	if (vdim.size() == 2) {
		beacls::FloatVec v;
		std::vector<beacls::FloatVec> vhist;
		getVelocity(v, vhist);
		h.resize(1);
		h[0] = std::atan2(v[1], v[0]);
		hhist.resize(vhist.size());
		std::transform(vhist.cbegin(), vhist.cend(), hhist.begin(), [](const auto& rhs) {
			return beacls::FloatVec{ std::atan2(rhs[1], rhs[0]) };
		});
		return true;
	}
	return false;
}
bool DynSys::getPosition(beacls::FloatVec& p, std::vector<beacls::FloatVec>& phist) const {
	if (!pimpl) return false;
	beacls::IntegerVec pdim = pimpl->get_pdim();
	beacls::FloatVec x = pimpl->get_x();
	std::vector<beacls::FloatVec> xhist = pimpl->get_xhist();
	p.resize(pdim.size());
	std::transform(pdim.cbegin(), pdim.cend(), p.begin(), [&x](const auto& rhs) { return x[rhs]; });
	phist.resize(xhist.size());
	std::transform(xhist.cbegin(), xhist.cend(), phist.begin(), [&pdim](const auto& xhist_i) {
		beacls::FloatVec x(pdim.size());
		std::transform(pdim.cbegin(), pdim.cend(), x.begin(), [&xhist_i](const auto& rhs) { return xhist_i[rhs]; });
		return x;
	});
	return true;
}
bool DynSys::getVelocity(beacls::FloatVec& v, std::vector<beacls::FloatVec>& vhist) const {
	if (!pimpl) return false;
	beacls::IntegerVec vdim = pimpl->get_vdim();
	if (!vdim.empty()) {
		beacls::FloatVec x = pimpl->get_x();
		std::vector<beacls::FloatVec> xhist = pimpl->get_xhist();
		v.resize(vdim.size());
		std::transform(vdim.cbegin(), vdim.cend(), v.begin(), [&x](const auto& rhs) {return x[rhs]; });
		vhist.resize(xhist.size());
		std::transform(xhist.cbegin(), xhist.cend(), vhist.begin(), [&vdim](const auto& xhixt_i) {
			std::vector <FLOAT_TYPE> vhist_i(vdim.size());
			std::transform(vdim.cbegin(), vdim.cend(), vhist_i.begin(), [&xhixt_i](const auto& rhs) {return xhixt_i[rhs]; });
			return vhist_i;
		});
	}
	return true;
}

bool DynSys_impl::plotPosition(const DynSys* dynSys, helperOC::PlotExtraArgs& extraArgs) {
	beacls::FloatVec p;
	std::vector<beacls::FloatVec> phist;
	dynSys->getPosition(p, phist);

	beacls::FloatVec v;
	std::vector<beacls::FloatVec> vhist;
	dynSys->getVelocity(v, vhist);

	const FLOAT_TYPE small = (FLOAT_TYPE)1e-2;
	const FLOAT_TYPE normv = std::sqrt(v[0] * v[0] + v[1] * v[1]);
	if (normv > small) {
		std::transform(v.cbegin(), v.cend(), v.begin(), [normv](const auto& rhs) { return rhs / normv; });
	}
	//!< Plot position trajectory
	if (hpxpyhist.empty()) {
		//!< If no graphics handle has been created, create it.
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;
	}
	else {
		//!< Otherwise, simply update the graphics handles
		hpxpyhist.resize(phist.size());
		for (size_t i = 0; i < phist.size(); ++i) {
			hpxpyhist[i].XData = phist[0];
			hpxpyhist[i].YData = phist[1];
		}
	}

	//!< Plot current position and velocity using an arrow
	if (hpxpy.XData.empty()) {
		//!< If no graphics handle has been created, create it with the specified
		//!< color.Use default color if no color is provided.
		//!<			obj.hpxpy = quiver(p(1), p(2), v(1), v(2), 'ShowArrowHead', ...
		//!<				'on', 'AutoScaleFactor', arrowLength);
		//!<	hold on
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;

		hpxpy.Marker = '.';
		hpxpy.Color = hpxpyhist[0].Color;
		hpxpy.MarkerFaceColor = hpxpyhist[0].Color;
		hpxpy.MarkerSize = extraArgs.MarkerSize;
		hpxpy.MaxHeadSize = 1;
		hpxpy.LineWidth = 1.5;
	}
	else {
		//!< Otherwise, simply update graphics handles
		hpxpy.XData.resize(1);
		hpxpy.YData.resize(1);
		hpxpy.XData[0] = p[0];
		hpxpy.YData[0] = p[1];
		hpxpy.UData = v[0];
		hpxpy.VData = v[1];
	}
	return true;
}

bool DynSys::plotPosition(helperOC::PlotExtraArgs& extraArgs) {
	if (pimpl) return pimpl->plotPosition(this, extraArgs);
	else return false;
}

bool DynSys_impl::updateState(
	beacls::FloatVec& x1,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& src_u,
	const FLOAT_TYPE T,
	const beacls::FloatVec& x0,
	const std::vector<beacls::FloatVec >& src_d
) {
	//!< If no state is specified, use current state
	const beacls::FloatVec& modified_x0 = (x0.empty()) ? x : x0;
	//!< If time horizon is 0, return initial state
	if (T == 0) {
		x1 = modified_x0;
		return true;
	}
	//!< Do nothing if control is empty
	if (src_u.empty()) {
		x1 = modified_x0;
		return true;
	}
	//!< Do nothing if control is not a number
	if (any_of(src_u.cbegin(), src_u.cend(), [](const auto& rhs) { 
		return (any_of(rhs.cbegin(), rhs.cend(), [](const auto& rhs) { return rhs == std::numeric_limits<FLOAT_TYPE>::signaling_NaN(); }));
	})) {
		std::cerr << "u = NaN" << std::endl;
		x1 = modified_x0;
		return true;
	}
	//!< Make sure control input is valid
	//!< T.B.D.

	//!< Check whether there's disturbance (this is needed since not all vehicle classes have dynamics that can handle disturbance)
	ode113(x1, dynSys, src_u, src_d, beacls::FloatVec{0, T}, x0);
	//!< Update the state, state history, control, and control history
	x = x1;
	u = src_u;
	xhist.push_back(x);
	uhist.push_back(u);
	return true;
}

size_t DynSys::find_val(const beacls::IntegerVec& vec, const size_t value) const {
  return std::find(vec.cbegin(), vec.cend(), value) - vec.cbegin();
}

bool DynSys::updateState(
	beacls::FloatVec& x1,
	const std::vector<beacls::FloatVec >& u,
	const FLOAT_TYPE T,
	const beacls::FloatVec& x0,
	const std::vector<beacls::FloatVec >& d
) {
	if (pimpl) return pimpl->updateState(x1, this, u, T, x0, d);
	else return false;
}

bool DynSys::optCtrl(
	std::vector<beacls::FloatVec >&,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>&,
	const beacls::IntegerVec&,
	const beacls::IntegerVec&,
	const DynSys_UMode_Type
) const {
	return true;
}

bool DynSys::optDstb(
	std::vector<beacls::FloatVec >& ,
	const FLOAT_TYPE ,
	const std::vector<beacls::FloatVec::const_iterator >& ,
	const std::vector<const FLOAT_TYPE* >&,
	const beacls::IntegerVec&,
	const beacls::IntegerVec&,
	const DynSys_DMode_Type
) const {
	return true;
}

bool DynSys::TIdyn(
	std::vector<beacls::FloatVec >&,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<beacls::FloatVec >&,
	const std::vector<beacls::FloatVec >&,
	const beacls::IntegerVec&,
	const size_t
) const {
	return true;
}

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool DynSys::optCtrl_cuda(
	std::vector<beacls::UVec>&,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const DynSys_UMode_Type
) const {
	return false;
}

bool DynSys::optDstb_cuda(
	std::vector<beacls::UVec>&,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const DynSys_DMode_Type
) const {
	return false;
}
bool DynSys::dynamics_cuda(
	std::vector<beacls::UVec>&,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t
) const {
	return false;
}
bool DynSys::optCtrl_cuda(
	std::vector<beacls::UVec>&,
	std::vector<beacls::UVec>&,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const DynSys_UMode_Type
) const {
	return false;
}
bool DynSys::optDstb_cuda(
	std::vector<beacls::UVec>&,
	std::vector<beacls::UVec>&,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const DynSys_DMode_Type
) const {
	return false;
}
bool DynSys::dynamics_cuda(
	beacls::UVec&,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t
) const {
	return false;
}
bool DynSys::TIdyn_cuda(
	std::vector<beacls::UVec>&,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t
) const {
	return false;
}
bool DynSys::HamFunction_cuda(
	beacls::UVec& ,
	const DynSysSchemeData* ,
	const FLOAT_TYPE ,
	const beacls::UVec& ,
	const std::vector<beacls::UVec>& ,
	const std::vector<beacls::UVec>& ,
	const size_t ,
	const size_t,
	const bool
) const {
	return false;
}
bool DynSys::PartialFunction_cuda(
	beacls::UVec&,
	const DynSysSchemeData*,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t,
	const size_t,
	const size_t
) const {
	return false;
}

#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */

size_t DynSys::get_nx() const {
	if (pimpl) return pimpl->get_nx();
	return 0;
}
size_t DynSys::get_nu() const {
	if (pimpl) return pimpl->get_nu();
	return 0;
}
size_t DynSys::get_nd() const {
	if (pimpl) return pimpl->get_nd();
	return 0;
}

beacls::FloatVec DynSys::get_x() const {
	if (pimpl) return pimpl->get_x();
	return beacls::FloatVec();
}
std::vector<beacls::FloatVec> DynSys::get_u() const {
	if (pimpl) return pimpl->get_u();
	return std::vector<beacls::FloatVec>();
}

std::vector<beacls::FloatVec> DynSys::get_xhist() const {
	if (pimpl) return pimpl->get_xhist();
	return std::vector<beacls::FloatVec>();
}
std::vector<std::vector<beacls::FloatVec>> DynSys::get_uhist() const {
	if (pimpl) return pimpl->get_uhist();
	return std::vector<std::vector<beacls::FloatVec>>();
}

beacls::IntegerVec DynSys::get_pdim() const {
	if (pimpl) return pimpl->get_pdim();
	return beacls::IntegerVec();
}
beacls::IntegerVec DynSys::get_vdim() const {
	if (pimpl) return pimpl->get_vdim();
	return beacls::IntegerVec();
}
beacls::IntegerVec DynSys::get_hdim() const {
	if (pimpl) return pimpl->get_hdim();
	return beacls::IntegerVec();
}
beacls::IntegerVec DynSys::get_TIdims() const {
	if (pimpl) return pimpl->get_TIdims();
	return beacls::IntegerVec();
}

HPxPy DynSys::get_hpxpy() const {
	if (pimpl) return pimpl->get_hpxpy();
	return HPxPy();
}
std::vector<HPxPy> DynSys::get_hpxpyhist() const {
	if (pimpl) return pimpl->get_hpxpyhist();
	return std::vector<HPxPy>();
}
beacls::FloatVec DynSys::get_hvxvy() const {
	if (pimpl) return pimpl->get_hvxvy();
	return beacls::FloatVec();
}
std::vector<beacls::FloatVec> DynSys::get_hvxvyhist() const {
	if (pimpl) return pimpl->get_hvxvyhist();
	return std::vector<beacls::FloatVec>();
}

beacls::FloatVec DynSys::get_hpv() const {
	if (pimpl) return pimpl->get_hpv();
	return beacls::FloatVec();
}
std::vector<beacls::FloatVec> DynSys::get_hpvhist() const {
	if (pimpl) return pimpl->get_hpvhist();
	return std::vector<beacls::FloatVec>();
}

beacls::FloatVec DynSys::get_data() const {
	if (pimpl) return pimpl->get_data();
	return beacls::FloatVec();
}

const helperOC::PartialFunction_cuda* DynSys::get_partialFunction() const {
	if (pimpl) return pimpl->get_partialFunction();
	return NULL;
}
#if 0
void DynSys::set_nx(const size_t nx) {
	if (pimpl) pimpl->set_nx(nx);
}
void DynSys::set_nu(const size_t nu) {
	if (pimpl) pimpl->set_nu(nu);
}
void DynSys::set_nd(const size_t nd) {
	if (pimpl) pimpl->set_nd(nd);
}
#endif
void DynSys::set_x(const beacls::FloatVec& x) {
	if (pimpl) pimpl->set_x(x);
}
void DynSys::set_u(const std::vector<beacls::FloatVec>& u) {
	if (pimpl) pimpl->set_u(u);
}

void DynSys::set_xhist(const std::vector<beacls::FloatVec>& xhist) {
	if (pimpl) pimpl->set_xhist(xhist);
}
void DynSys::set_uhist(const std::vector<std::vector<beacls::FloatVec>>& uhist) {
	if (pimpl) pimpl->set_uhist(uhist);
}
#if 0
void DynSys::set_pdim(const beacls::IntegerVec& pdim) {
	if (pimpl) pimpl->set_pdim(pdim);
}
void DynSys::set_vdim(const beacls::IntegerVec& vdim) {
	if (pimpl) pimpl->set_vdim(vdim);
}
void DynSys::set_hdim(const beacls::IntegerVec& hdim) {
	if (pimpl) pimpl->set_hdim(hdim);
}
void DynSys::set_TIdims(const beacls::IntegerVec& TIdims) {
	if (pimpl) pimpl->set_TIdims(TIdims);
}
#endif
void DynSys::set_hpxpy(const HPxPy& hpxpy) {
	if (pimpl) pimpl->set_hpxpy(hpxpy);
}
void DynSys::set_hpxpyhist(const std::vector<HPxPy>& hpxpyhist) {
	if (pimpl) pimpl->set_hpxpyhist(hpxpyhist);
}
void DynSys::set_hvxvy(const beacls::FloatVec& hvxvy) {
	if (pimpl) pimpl->set_hvxvy(hvxvy);
}
void DynSys::set_hvxvyhist(const std::vector<beacls::FloatVec>& hvxvyhist) {
	if (pimpl) pimpl->set_hvxvyhist(hvxvyhist);
}

void DynSys::set_hpv(const beacls::FloatVec& hpv) {
	if (pimpl) pimpl->set_hpv(hpv);
}
void DynSys::set_hpvhist(const std::vector<beacls::FloatVec>& hpvhist) {
	if (pimpl) pimpl->set_hpvhist(hpvhist);
}

void DynSys::clear_x() {
	if (pimpl) pimpl->clear_x();
}
void DynSys::clear_u() {
	if (pimpl) pimpl->clear_u();
}
void DynSys::clear_xhist() {
	if (pimpl) pimpl->clear_xhist();
}
void DynSys::clear_uhist() {
	if (pimpl) pimpl->clear_uhist();
}


void DynSys::push_back_xhist(const beacls::FloatVec& x) {
	if (pimpl) pimpl->push_back_xhist(x);
}

void DynSys::set_data(const beacls::FloatVec& data) {
	if (pimpl) pimpl->set_data(data);
}
void DynSys::set_partialFunction(const helperOC::PartialFunction_cuda* partialFunction) {
	if (pimpl) pimpl->set_partialFunction(partialFunction);
}

DynSys_impl* DynSys_impl::clone() const {
	return new DynSys_impl(*this);
}
DynSys::DynSys(const DynSys& rhs) :
	pimpl(rhs.pimpl->clone())
{
}
