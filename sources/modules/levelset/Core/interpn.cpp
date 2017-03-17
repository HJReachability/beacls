#include <typedef.hpp>
#include <Core/interpn.hpp>
#include <vector>
#include <deque>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <cmath>

namespace beacls {
	static bool convertv(
		beacls::FloatVec& convertedV,
		int& origvtype,
		const beacls::FloatVec& V,
		const Interpolate_Type interp_method,
		const std::vector<const beacls::FloatVec*>& Xq
	);
	class GriddedInterpolant {
	public:
		std::vector<const beacls::FloatVec*> GridVectors;
		const beacls::FloatVec* Values;
		Interpolate_Type Method;
		std::vector<beacls::Extrapolate_Type> extrap_methods;
		GriddedInterpolant(
			const std::vector<const beacls::FloatVec*>& gv,
			const beacls::FloatVec* v,
			const Interpolate_Type m,
			const std::vector<beacls::Extrapolate_Type>& em
			) : GridVectors(gv), Values(v), Method(m), extrap_methods(em) {}
		bool operator()(
			beacls::FloatVec& V,
			std::vector<const beacls::FloatVec*>& Xq) const;
	};
	static bool compactgridformat(
		const std::vector<const beacls::FloatVec*>& Xq
	);

};
static bool beacls::compactgridformat(
	const std::vector<const beacls::FloatVec*>& Xq
) {
	bool tf = all_of(Xq.cbegin(), Xq.cend(), [](const auto& rhs) { return rhs->size() == 1; });
	if (tf && (Xq.size() > 1)) {
		std::deque<bool> ns(Xq.size());
		std::transform(Xq.cbegin(), Xq.cend(), ns.begin(), [](const auto& rhs) {
			return rhs->size() != 1;
		});
		bool ns0 = ns[0];
		tf = !all_of(ns.cbegin()+1, ns.cend(), [ns0](const auto& rhs) { return rhs == ns0; });
	}
	return tf;
}
bool beacls::GriddedInterpolant::operator()(
	beacls::FloatVec& V,
	std::vector<const beacls::FloatVec*>& Xq
	) const {

	if (Method != Interpolate_linear) {
		std::cerr << "Invalid method" << Method << std::endl;
		return false;
	}
	if (std::any_of(extrap_methods.cbegin(), extrap_methods.cend(), [](const auto& rhs) { 
		return (rhs != beacls::Extrapolate_none) && (rhs != beacls::Extrapolate_periodic) && (rhs != beacls::Extrapolate_nearest); })){
		std::cerr << "Invalid method" << Method << std::endl;
		return false;
	}
	const size_t num_of_dimensions = Xq.size();
	size_t last_val = Xq[0]->size();
	bool combinationAxis = false;
	for (size_t dimension = 1; dimension < num_of_dimensions; ++dimension) {
		if (Xq[dimension]->size() != last_val) combinationAxis = true;
	}
	size_t num_of_interpolate_points = combinationAxis 
		? std::accumulate(Xq.cbegin(), Xq.cend(), static_cast<size_t>(1), [](const auto& lhs, const auto& rhs) { return lhs * rhs->size(); }) 
		: Xq[0]->size();
	V.resize(num_of_interpolate_points);

	const size_t max_num_of_adjacents = static_cast<size_t>(std::pow(2, num_of_dimensions));
	beacls::IntegerVec lower_bounds(Xq.size());
	beacls::IntegerVec upper_bounds(Xq.size());
	beacls::FloatVec adjacent_values(max_num_of_adjacents);
	beacls::FloatVec interpolated_values(max_num_of_adjacents);

	beacls::IntegerVec inner_sizes(num_of_dimensions);
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		inner_sizes[dimension] = std::accumulate(GridVectors.cbegin(), GridVectors.cbegin() + dimension, static_cast<size_t>(1), [](const auto& lhs, const auto& rhs) { return lhs * rhs->size(); });
	}


	for (size_t interpolate_point = 0; interpolate_point < num_of_interpolate_points; ++interpolate_point) {
		if (combinationAxis) {
			for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
				const size_t inner_size = inner_sizes[dimension];
				const size_t index = (interpolate_point /inner_size) % GridVectors[dimension]->size();
				const FLOAT_TYPE val = (*Xq[dimension])[index];
				lower_bounds[dimension] = std::distance(GridVectors[dimension]->cbegin(), std::lower_bound(GridVectors[dimension]->cbegin(), GridVectors[dimension]->cend(), val));
				upper_bounds[dimension] = std::distance(GridVectors[dimension]->cbegin(), std::upper_bound(GridVectors[dimension]->cbegin(), GridVectors[dimension]->cend(), val));
			}
		}
		else {
			std::transform(GridVectors.cbegin(), GridVectors.cend(), Xq.cbegin(), lower_bounds.begin(), [interpolate_point](const auto& lhs, const auto& rhs) {
				const size_t index = interpolate_point;
				const FLOAT_TYPE val = (*rhs)[index];
				return std::distance(lhs->cbegin(), std::lower_bound(lhs->cbegin(), lhs->cend(), val));
			});
			std::transform(GridVectors.cbegin(), GridVectors.cend(), Xq.cbegin(), upper_bounds.begin(), [interpolate_point](const auto& lhs, const auto& rhs) {
				const size_t index = interpolate_point;
				const FLOAT_TYPE val = (*rhs)[index];
				return std::distance(lhs->cbegin(), std::upper_bound(lhs->cbegin(), lhs->cend(), val));
			});
		}
		bool out_of_range = false;
		for (size_t index = 0; index < adjacent_values.size(); ++index) {
			size_t src_index = 0;
			size_t num_of_inner_dimension_elements = 1;
			for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
				const beacls::Extrapolate_Type extrapolate_method = extrap_methods[dimension];
				bool flag = ((index >> dimension) & 0x1);
				const size_t dimension_flag = flag ? 1 : 0;
				const size_t lower_bounds_index = lower_bounds[dimension];
				const size_t upper_bounds_index = upper_bounds[dimension];
				const size_t grid_size = GridVectors[dimension]->size();
				if (lower_bounds_index != upper_bounds_index) {
					if (lower_bounds_index >= grid_size) { //!< On line
						switch (extrapolate_method) {
						default:
						case beacls::Extrapolate_none:
							out_of_range = true;
							break;
						case beacls::Extrapolate_nearest:
							src_index += (grid_size - 1)* num_of_inner_dimension_elements;
							break;
						case beacls::Extrapolate_periodic:
							src_index += (lower_bounds_index%grid_size)* num_of_inner_dimension_elements;
							break;
						}
					}
					else {
						src_index += (lower_bounds_index)* num_of_inner_dimension_elements;
					}
				}
				else {	//!< Between lines
					if (lower_bounds_index == 0) {
						switch (extrapolate_method) {
						default:
						case beacls::Extrapolate_none:
							out_of_range = true;
							break;
						case beacls::Extrapolate_nearest:
							break;
						case beacls::Extrapolate_periodic:
							if (dimension_flag == 0)
								src_index += (grid_size - 1) * num_of_inner_dimension_elements;
							break;
						}
					}
					else if (lower_bounds_index >= grid_size) {
						switch (extrapolate_method) {
						default:
						case beacls::Extrapolate_none:
							out_of_range = true;
							break;
						case beacls::Extrapolate_nearest:
							src_index += (grid_size - 1) * num_of_inner_dimension_elements;
							break;
						case beacls::Extrapolate_periodic:
							src_index += (lower_bounds_index - 1 + dimension_flag) % grid_size * num_of_inner_dimension_elements;
							break;
						}
					}
					else {
						src_index += (lower_bounds_index - 1 + dimension_flag) * num_of_inner_dimension_elements;
					}
				}
				num_of_inner_dimension_elements *= GridVectors[dimension]->size();
			}
			if (!out_of_range) {
				adjacent_values[index] = (*Values)[src_index];
			}
			else {
				break;
			}
		}
		if (!out_of_range) {
			size_t num_of_interpolated_values = adjacent_values.size();
			for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
				const beacls::Extrapolate_Type extrapolate_method = extrap_methods[dimension];
				num_of_interpolated_values /= 2;
				FLOAT_TYPE weight;
				size_t index = 0;
				if (combinationAxis) {
					const size_t inner_size = inner_sizes[dimension];
					index = (interpolate_point / inner_size) % Xq[dimension]->size();
				}
				else {
					index = interpolate_point;
				}
				const FLOAT_TYPE point = (*Xq[dimension])[index];
				const size_t lower_bounds_index = lower_bounds[dimension];
				const size_t upper_bounds_index = upper_bounds[dimension];
				const size_t grid_size = GridVectors[dimension]->size();


				if (lower_bounds_index == upper_bounds_index) {	//!< Between lines
					switch (extrapolate_method) {
					default:
					case beacls::Extrapolate_none:
					case beacls::Extrapolate_nearest:
						if (lower_bounds_index <= 0) {
							weight = 1;
						}
						else if (upper_bounds_index >= grid_size) {
							weight = 0;
						}
						else {
							const FLOAT_TYPE left_point = (*GridVectors[dimension])[lower_bounds_index - 1];
							const FLOAT_TYPE right_point = (*GridVectors[dimension])[lower_bounds_index];
							weight = (right_point - point) / (right_point - left_point);
						}
						break;
					case beacls::Extrapolate_periodic:
						{
							const size_t left_index = (lower_bounds_index - 1 + grid_size) % grid_size;
							const size_t right_index = (lower_bounds_index + grid_size) % grid_size;

							const FLOAT_TYPE left_point = (*GridVectors[dimension])[left_index];
							const FLOAT_TYPE right_point = (*GridVectors[dimension])[right_index];
							if (right_point > point) {
								if (right_point > left_point) {
									weight = (right_point - point) / (right_point - left_point);
								}
								else {
									const FLOAT_TYPE delta = (*GridVectors[dimension])[1] -(*GridVectors[dimension])[0];
									const FLOAT_TYPE max_point = (*GridVectors[dimension])[grid_size - 1] + delta;
									weight = (right_point - point) / (right_point + max_point - left_point);
								}
							}
							else {
								if (right_point > left_point) {
									const FLOAT_TYPE delta = (*GridVectors[dimension])[1] -(*GridVectors[dimension])[0];
									const FLOAT_TYPE max_point = (*GridVectors[dimension])[grid_size - 1] + delta;
									weight = (right_point + max_point - point) / (right_point - left_point);
								}
								else {
									const FLOAT_TYPE delta = (*GridVectors[dimension])[1] -(*GridVectors[dimension])[0];
									const FLOAT_TYPE max_point = (*GridVectors[dimension])[grid_size - 1] + delta;
									weight = (right_point + max_point - point) / (right_point + max_point - left_point);
								}
							}
						}
						break;
					}


				}
				else {	//!< On line
					weight = 1;
				}
				
				std::max<FLOAT_TYPE>(std::min<FLOAT_TYPE>(weight, 1) , 0);
				for (size_t i = 0; i < num_of_interpolated_values; ++i) {
					const FLOAT_TYPE left_value = adjacent_values[i * 2];
					const FLOAT_TYPE right_value = adjacent_values[i * 2 + 1];
					const FLOAT_TYPE interpolated_value = left_value * weight + right_value * (1 - weight);
					interpolated_values[i] = interpolated_value;
				}
				adjacent_values = interpolated_values;
			}
			V[interpolate_point] = interpolated_values[0];
		}
		else {
			V[interpolate_point] = std::numeric_limits<FLOAT_TYPE>::quiet_NaN();
		}
	}
	return true;
}

static bool beacls::convertv(
	beacls::FloatVec& ,
	int& ,
	const beacls::FloatVec& ,
	const Interpolate_Type ,
	const std::vector<const beacls::FloatVec*>& 
	/*
	beacls::FloatVec& convertedV,
	int& origvtype,
	const beacls::FloatVec& V,
	const Interpolate_Type interp_method,
	const std::vector<const beacls::FloatVec*>& Xq
	*/
) {
//	std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;
	return true;
}

bool beacls::interpn(
	beacls::FloatVec& Vq,
	const std::vector<const beacls::FloatVec*>& X_ptrs,
	const std::vector<beacls::IntegerVec>& Ns,
	const Interpolate_Type interp_method,
	const std::vector<beacls::Extrapolate_Type>& extrap_methods
) {
	const bool isspline = (interp_method == Interpolate_spline);
	std::vector<beacls::Extrapolate_Type> modified_extrap_methods;
	const size_t nargs = X_ptrs.size();
	const size_t mididx = static_cast<size_t>(std::ceil(static_cast<FLOAT_TYPE>(nargs) / 2)) - 1;
	if (extrap_methods.empty()) {
		beacls::Extrapolate_Type extrap_method = (isspline) ? Extrapolate_spline : Extrapolate_none;
		modified_extrap_methods.resize(mididx, extrap_method);
	}
	else {
		modified_extrap_methods = extrap_methods;
	}
	GriddedInterpolant* F_ptr = NULL;
	std::vector<const beacls::FloatVec*> Xq;
	std::vector<beacls::FloatVec> reduced_X;
	std::vector<const beacls::FloatVec* > reduced_X_ptrs;
	if (nargs == 1) {
		//!< T.B.D.
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;
	}
	else {
		const size_t ndimsarg1 = Ns[0].size();
		if ((ndimsarg1 != 1) && (nargs == (ndimsarg1 + 1))) {
			//!< interpn(V, X1q, X2q, X3q,...)
			//!< T.B.D.
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;
		}
		else if (nargs % 2 == 1) {
			//!< interpn(X1, X2, X3, ..., V, X1q, X2q, X3q,...)
			std::vector<const beacls::FloatVec*> X(mididx);
			std::vector<beacls::IntegerVec > XNs(mididx);
			Xq.resize(mididx);
			std::copy(X_ptrs.cbegin(), X_ptrs.cbegin() + mididx, X.begin());
			std::copy(Ns.cbegin(), Ns.cbegin() + mididx, XNs.begin());
			std::copy(X_ptrs.cbegin()+mididx+1, X_ptrs.cend(), Xq.begin());
			const beacls::FloatVec* V_ptr = X_ptrs[mididx];
			int origvtype;
			beacls::FloatVec convertedV;
			convertv(convertedV, origvtype, *V_ptr, interp_method, Xq);
			if (isspline) {
				//!< T.B.D.
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;
			}
#if 0
			beacls::FloatVec checkedV;
			checkmonotonic(X, checkedV, X, *V_ptr);
			V_ptr = &checkedV;
#endif
			if (std::all_of(XNs.cbegin(), XNs.cend(), [](const auto& rhs) { return rhs.size() == 1; })) {
				F_ptr = new GriddedInterpolant(X, V_ptr, interp_method, modified_extrap_methods);
			}
			else {
				reduced_X.resize(X.size());
				reduced_X_ptrs.resize(X.size());
				for (size_t dimension = 0; dimension < XNs.size(); ++dimension) {
					reduced_X[dimension].resize(XNs[dimension][dimension]);
					const size_t inner_loop_size = std::accumulate(XNs[dimension].cbegin() + 1, XNs[dimension].cbegin() + 1 + dimension, static_cast<size_t>(1), [](const auto& lhs, const auto& rhs) {
						return lhs * rhs;
					});
					for (size_t index = 0; index < reduced_X[dimension].size(); ++index) {
						reduced_X[dimension][index] = (*X[dimension])[index*inner_loop_size];
					}
					reduced_X_ptrs[dimension] = &reduced_X[dimension];
				}
				F_ptr = new GriddedInterpolant(reduced_X_ptrs, V_ptr, interp_method, modified_extrap_methods);
			}
		}
		else {
			std::cerr << "Error! " << __func__ << ": Invalid  nargin: " << nargs << std::endl;
			return false;
		}

	}
	if ((interp_method == Interpolate_cubic) && (F_ptr->Method == Interpolate_spline)) {
		//!< T.B.D.
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;
	}

	//!< Now interpolate
	bool iscompact = compactgridformat(Xq);
	bool result = false;
	if (iscompact || ((F_ptr->Method == Interpolate_spline) && (Xq[0]->size() == 1))) {
		// Vq = F(Xq)
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " is not implemented yet!" << std::endl;
	}
	else {
		//!< Check overwriting
		if (std::all_of(X_ptrs.cbegin(), X_ptrs.cend(), [&Vq](const auto& rhs) { return rhs != &Vq; })) {
			result = F_ptr->operator()(Vq, Xq);
		}
		else {
			beacls::FloatVec tmp_Vq;
			result = F_ptr->operator()(tmp_Vq, Xq);
			Vq = tmp_Vq;
		}
	}

	if (F_ptr) delete F_ptr;

	return result;
}
