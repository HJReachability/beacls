#ifndef __HJI_Grid_impl_hpp__
#define __HJI_Grid_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <levelset/BoundaryCondition/AddGhostPeriodic.hpp>

namespace levelset {
	class BoundaryCondition;
	typedef AddGhostPeriodic defaultBoundaryCondition;

	static const FLOAT_TYPE defaultMin = 0.;
	static const FLOAT_TYPE defaultMax = 1.;
	static const size_t defaultN = 101;
	//! This is just to avoid attempts to allocate 100 dimensional arrays.
	static const size_t maxDimension = 5;
	static const std::vector<FLOAT_TYPE> defaultBoundaryData;

	class HJI_Grid_impl {
	private:
		size_t num_of_dimensions;
		std::vector<FLOAT_TYPE> mins;
		std::vector<FLOAT_TYPE> maxs;
		std::vector<BoundaryCondition*> boundaryConditions;
		std::vector<size_t> Ns;
		std::vector<FLOAT_TYPE> dxs;
		std::vector<FLOAT_TYPE> dxInvs;
		std::vector<std::vector<FLOAT_TYPE> > vss;
		mutable beacls::UVec v_uvec;
		std::vector<std::vector<FLOAT_TYPE> > xss;
		std::vector<FLOAT_TYPE> axis;
		std::vector<size_t> shape;

		typedef enum MATFile_Data_Type {
			MATFILE_Data_Type_Invalid = 0,	//!< Invalid
			MATFILE_Data_Type_miINT8 = 1,	//!< miINT8
			MATFILE_Data_Type_miUINT8 = 2,	//!< miUINT8
			MATFILE_Data_Type_miINT16 = 3,	//!< miINT16
			MATFILE_Data_Type_miUINT16 = 4,	//!< miUINT16
			MATFILE_Data_Type_miINT32 = 5,	//!< miINT32
			MATFILE_Data_Type_miUINT32 = 6,	//!< miUINT32
			MATFILE_Data_Type_miSINGLE = 7,	//!< miSINGLE
			MATFILE_Data_Type_Reserved8 = 8,	//!< Reserved
			MATFILE_Data_Type_miDOUBLE = 9,	//!< miDOUBLE
			MATFILE_Data_Type_Reserved10 = 10,	//!< Reserved
			MATFILE_Data_Type_Reserved11 = 11,	//!< Reserved
			MATFILE_Data_Type_miINT64 = 12,	//!< miINT64
			MATFILE_Data_Type_miUINT64 = 13,	//!< miUINT64
			MATFILE_Data_Type_miMATRIX = 14,	//!< miMATRIX
			MATFILE_Data_Type_miCOMPRESSED = 15,	//!< miCOMPRESSED
			MATFILE_Data_Type_miUTF8 = 16,	//!< miUTF8
			MATFILE_Data_Type_miUTF16 = 17,	//!< miUTF16
			MATFILE_Data_Type_miUTF32 = 18,	//!< miUTF32
		}MATFile_Data_Type;
		typedef enum MATLAB_Array_Type {
			MATLAB_Array_Type_Invalid = 0,	//!< Invalid
			MATLAB_Array_Type_mxCELL_CLASS = 1,	//!< mxCELL_CLASS
			MATLAB_Array_Type_mxSTRUCT_CLASS = 2,	//!< mxSTRUCT_CLASS
			MATLAB_Array_Type_mxOBJECT_CLASS = 3,	//!< mxOBJECT_CLASS
			MATLAB_Array_Type_mxCHAR_CLASS = 4,	//!< mxCHAR_CLASS
			MATLAB_Array_Type_mxSPARSE_CLASS = 5,	//!< mxSPARSE_CLASS
			MATLAB_Array_Type_mxDOUBLE_CLASS = 6,	//!< mxDOUBLE_CLASS
			MATLAB_Array_Type_mxSINGLE_CLASS = 7,	//!< mxSINGLE_CLASS
			MATLAB_Array_Type_mxINT8_CLASS = 8,	//!< mxINT8_CLASS
			MATLAB_Array_Type_mxUINT8_CLASS = 9,	//!< mxUINT8_CLASS
			MATLAB_Array_Type_mxINT16_CLASS = 10,	//!< mxINT16_CLASS
			MATLAB_Array_Type_mxUINT16_CLASS = 11,	//!< mxUINT16_CLASS
			MATLAB_Array_Type_mxINT32_CLASS = 12,	//!< mxINT32_CLASS
			MATLAB_Array_Type_mxUINT32_CLASS = 13,	//!< mxUINT32_CLASS
			MATLAB_Array_Type_mxINT64_CLASS = 14,	//!< mxINT64_CLASS
			MATLAB_Array_Type_mxUINT64_CLASS = 15,	//!< mxUINT64_CLASS
		}MATLAB_Array_Type;
	public:
		HJI_Grid_impl(
		);
		HJI_Grid_impl(
			const size_t num_of_dimensions
		);
		~HJI_Grid_impl();
		bool operator==(const HJI_Grid_impl& rhs) const;
		size_t get_numel() const {
			size_t num_of_elems = 1;
			for (size_t i = 0; i < num_of_dimensions; ++i) {
				num_of_elems *= Ns[i];
			}
			return num_of_elems;
		}

		size_t get_sum_of_elems() const {
			size_t num_of_elems = 1;
			for (size_t i = 0; i < num_of_dimensions; ++i) {
				num_of_elems *= Ns[i];
			}
			return num_of_elems;
		}
		
		void set_num_of_dimensions(size_t a) { num_of_dimensions = a; }
		void set_mins(const std::vector<FLOAT_TYPE>& a) { mins = a; }
		void set_maxs(const std::vector<FLOAT_TYPE>& a) { maxs = a; }
		void set_boundaryConditions(const std::vector<BoundaryCondition*>& a) {
			for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](auto &ptr) {if (ptr)delete ptr; }));
			this->boundaryConditions.resize(a.size());
			std::transform(a.cbegin(), a.cend(), boundaryConditions.begin(), ([](auto &ptr) { return ptr->clone(); }));
		}
		void set_Ns(const std::vector<size_t>& a) { Ns = a; }
		void set_dxInvs() {
			dxInvs.resize(dxs.size());
			std::transform(dxs.cbegin(), dxs.cend(), dxInvs.begin(), ([](const auto &rhs) {return 1. / rhs; }));
		}
		void set_dxInvs(const std::vector<FLOAT_TYPE>& a) {
			dxInvs = a;
		}
		void set_dxs(const std::vector<FLOAT_TYPE>& a) {
			dxs = a;
			set_dxInvs();
		}
		void set_vss(const std::vector<std::vector<FLOAT_TYPE> >& a) { vss = a; }
		void set_xss(const std::vector<std::vector<FLOAT_TYPE> >& a) { xss = a; }
		void set_axis(const std::vector<FLOAT_TYPE>& a) { axis = a; }
		void set_shape(const std::vector<size_t>& a) { shape = a; }


		size_t get_num_of_dimensions() const { return num_of_dimensions; }
		const std::vector<FLOAT_TYPE>& get_mins() const { return mins; }
		const std::vector<FLOAT_TYPE>& get_maxs() const { return maxs; }
		BoundaryCondition* get_boundaryCondition(const size_t dimension) const { return boundaryConditions[dimension]; }
		const std::vector<size_t>& get_Ns() const { return Ns; }
		size_t get_N(const size_t dimension) const { return Ns[dimension]; }
		const std::vector<FLOAT_TYPE>& get_dxs() const { return dxs; }
		const std::vector<FLOAT_TYPE>& get_dxInvs() const { return dxInvs; }
		FLOAT_TYPE get_dx(const size_t dimension) const { return dxs[dimension]; }
		FLOAT_TYPE get_dxInv(const size_t dimension) const { return dxInvs[dimension]; }
		const std::vector<std::vector<FLOAT_TYPE> >& get_vss() const { return vss; }
		const std::vector<std::vector<FLOAT_TYPE> >& get_xss() const { return xss; }
		const std::vector<FLOAT_TYPE>& get_vs(const size_t dimension) const { return vss[dimension]; }
		const std::vector<FLOAT_TYPE>& get_xs(const size_t dimension) const { return xss[dimension]; }
		void calc_xs(
			beacls::UVec& x_uvec,
			const size_t dimension,
			const size_t start_index,
			const size_t length) const;
		void get_xs(
			beacls::UVec& x_uvec,
			const size_t dimension,
			const size_t start_index,
			const size_t length) const;
		void get_xss(
			std::vector<beacls::UVec>&  x_uvecs,
			const size_t start_index,
			const size_t length) const;
		const std::vector<FLOAT_TYPE>& get_axis() const { return axis; }
		const std::vector<size_t>& get_shape() const { return shape; }

		bool save_grid(
			const std::string &grid_name,
			beacls::MatFStream* fs,
			beacls::MatVariable* parent,
			const size_t cell_index,
			const bool compress) const;
		bool load_grid(
			const std::string &grid_name,
			beacls::MatFStream* fs,
			beacls::MatVariable* parent,
			const size_t cell_index);


		HJI_Grid_impl* clone(const bool cloneAll) const {
			if (cloneAll) {
				return new HJI_Grid_impl(*this);
			}
			else {	//!< clone except vss and xss.
				HJI_Grid_impl* g = new HJI_Grid_impl(num_of_dimensions);
				g->mins = mins;
				g->maxs = maxs;
				g->boundaryConditions.resize(boundaryConditions.size());
				std::transform(boundaryConditions.cbegin(), boundaryConditions.cend(), g->boundaryConditions.begin(), ([](auto &ptr) { return ptr->clone(); }));
				g->Ns = Ns;
				g->dxs = dxs;
				g->dxInvs = dxInvs;
				g->axis = axis;
				g->shape = shape;
				return g;
			}
		};

		bool processGrid(const std::vector<FLOAT_TYPE> &data);
	private:
		/** @overload
		Disable operator=
		*/
		HJI_Grid_impl& operator=(const HJI_Grid_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		HJI_Grid_impl(const HJI_Grid_impl& rhs);
	};
};
#endif	/* __HJI_Grid_impl_hpp__ */

