#ifndef __SchemeData_impl_hpp__
#define __SchemeData_impl_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <algorithm>
#include <typedef.hpp>
#include <cmath>
#include <cstring>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
#include <levelset/SpatialDerivative/SpatialDerivative.hpp>
#include <levelset/ExplicitIntegration/Dissipations/Dissipation.hpp>
#include <levelset/ExplicitIntegration/Terms/Term.hpp>
namespace levelset {
	class HJI_Grid;
	class SchemeData_impl {
	public:
	private:
		const HJI_Grid* grid;
		SpatialDerivative* spatialDerivative;
		Dissipation* dissipation;
		const Term* innerFunc;
		const SchemeData* innerData;
		bool positive;
	public:
		SchemeData_impl(
		) :
			grid(NULL),
			spatialDerivative(NULL),
			dissipation(NULL),
			innerFunc(NULL),
			innerData(NULL),
			positive(false) {}
		~SchemeData_impl() {
			if (spatialDerivative) delete spatialDerivative;
			if (dissipation) delete dissipation;
			if (innerFunc) delete innerFunc;
			if (innerData) delete innerData;
		}
		PREFIX_VC_DLL
			bool operator==(const SchemeData_impl& rhs) const;

		const HJI_Grid* get_grid() const {
			if (grid)
				return grid;
			else if (innerData)
				return innerData->get_grid();
			return NULL;
		}
		SpatialDerivative* get_spatialDerivative() const {
			if (spatialDerivative)
				return spatialDerivative;
			else if (innerData)
				return innerData->get_spatialDerivative();
			return NULL;
		}
		Dissipation* get_dissipation() const {
			if (dissipation)
				return dissipation;
			else if (innerData)
				return innerData->get_dissipation();
			return NULL;
		}
		const Term* get_innerFunc() const {
			if (innerFunc)
				return innerFunc;
			else if (innerData)
				return innerData->get_innerFunc();
			return NULL;
		}
		const SchemeData* get_innerData() const { return innerData; }
		bool get_positive() const { return positive; }

		void set_grid(const HJI_Grid* val) { grid = val; }
		void set_spatialDerivative(SpatialDerivative* val) { spatialDerivative = val->clone(); };
		void set_dissipation(Dissipation* val) { dissipation = val->clone(); };
		void set_innerFunc(const Term* val) { innerFunc = val->clone(); };
		void set_innerData(const SchemeData* val) { innerData = val->clone(); };
		void set_positive(const bool val) { positive = val; };

		SchemeData_impl* clone() const {
			return new SchemeData_impl(*this);
		};
	private:

		/** @overload
		Disable operator=
		*/
		SchemeData_impl& operator=(const SchemeData_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		SchemeData_impl(const SchemeData_impl& rhs) :
			grid(rhs.grid),
			spatialDerivative(rhs.spatialDerivative ? rhs.spatialDerivative->clone() : NULL),
			dissipation(rhs.dissipation ? rhs.dissipation->clone() : NULL),
			innerFunc(rhs.innerFunc ? rhs.innerFunc->clone() : NULL),
			innerData(rhs.innerData ? rhs.innerData->clone() : NULL),
			positive(rhs.positive)
		{}
	};
};
#endif	/* __SchemeData_impl_hpp__ */

