#ifndef __HJI_Grid_hpp__
#define __HJI_Grid_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <deque>
#include <algorithm>
#include <typedef.hpp>
#include <cmath>
#include <iostream>
#include <cstring>
#include <utility>
#include <Core/UVec.hpp>
using namespace std::rel_ops;


namespace beacls {
	PREFIX_VC_DLL
		MatFStream* openMatFStream(
			const std::string& file_name,
			const MatOpenMode mode
		);
	PREFIX_VC_DLL
		bool closeMatFStream(
			MatFStream* fs
		);
	PREFIX_VC_DLL
		MatVariable* openMatVariable(
			MatFStream* fs,
			const std::string& variable_name
		);
	PREFIX_VC_DLL
		bool closeMatVariable(
			MatVariable* variable
		);
	PREFIX_VC_DLL
		MatVariable* createMatStruct(
			const std::string& variable_name
		);
	PREFIX_VC_DLL
		bool writeMatVariable(
			MatFStream* fs,
			MatVariable* variable,
			const bool compress = true);
	PREFIX_VC_DLL
		MatVariable* createMatCell(
			const std::string& variable_name,
			const size_t size
		);
	PREFIX_VC_DLL
		bool setVariableToStruct(
			MatVariable* parent,
			MatVariable* child,
			const std::string& field_name);
	PREFIX_VC_DLL
		bool setVariableToCell(
			MatVariable* parent,
			MatVariable* child,
			const size_t index);
	PREFIX_VC_DLL
		MatVariable* getVariableFromStruct(
			MatVariable* parent,
			const std::string& variable_name
		);
	PREFIX_VC_DLL
		MatVariable* getVariableFromCell(
			MatVariable* parent,
			const size_t cell_index);
	PREFIX_VC_DLL
		size_t getCellSize(
			MatVariable* variable_ptr);
};
namespace levelset {
	class BoundaryCondition;

	class HJI_Grid_impl;

	class HJI_Grid {
	public:
	private:
		HJI_Grid_impl *pimpl;
		const std::vector<FLOAT_TYPE> dummy_float_type_vector;	//! dummy result for if pimpl is invalid;
		const std::vector<std::vector<FLOAT_TYPE> > dummy_float_type_vector_vector;	//! dummy result for if pimpl is invalid;
		const std::vector<size_t> dummy_size_t_vector;
	public:

		PREFIX_VC_DLL
			HJI_Grid(
			);
		PREFIX_VC_DLL
			HJI_Grid(
				const size_t num_of_dimensions
			);
		HJI_Grid(HJI_Grid_impl* pimpl);
		PREFIX_VC_DLL
			~HJI_Grid();
		PREFIX_VC_DLL
			bool operator==(const HJI_Grid& rhs) const;
		PREFIX_VC_DLL
			size_t get_sum_of_elems() const;
		PREFIX_VC_DLL
			size_t get_num_of_dimensions() const;

		PREFIX_VC_DLL
			void set_num_of_dimensions(size_t num_of_dimensions);
		PREFIX_VC_DLL
			void set_mins(const std::vector<FLOAT_TYPE>& mins);
		PREFIX_VC_DLL
			void set_maxs(const std::vector<FLOAT_TYPE>& maxs);
		PREFIX_VC_DLL
			void set_boundaryConditions(const std::vector<BoundaryCondition*>& boundaryConditions);
		PREFIX_VC_DLL
			void set_Ns(const std::vector<size_t>& Ns);
		PREFIX_VC_DLL
			void set_dxs(const std::vector<FLOAT_TYPE>& dxs);
		PREFIX_VC_DLL
			void set_vss(const std::vector<std::vector<FLOAT_TYPE> >& vss);
		PREFIX_VC_DLL
			void set_xss(const std::vector<std::vector<FLOAT_TYPE> >& xss);
		PREFIX_VC_DLL
			void set_axis(const std::vector<FLOAT_TYPE>& axis);
		PREFIX_VC_DLL
			void set_shape(const std::vector<size_t>& shape);

		PREFIX_VC_DLL
			const std::vector<FLOAT_TYPE>& get_mins() const;
		PREFIX_VC_DLL
			const std::vector<FLOAT_TYPE>& get_maxs() const;
		PREFIX_VC_DLL
			BoundaryCondition* get_boundaryCondition(const size_t dimension) const;
		PREFIX_VC_DLL
			const std::vector<size_t>& get_Ns() const;
		PREFIX_VC_DLL
			size_t get_N(const size_t dimension) const;
		const std::vector<FLOAT_TYPE>& get_dxs() const;
		const std::vector<FLOAT_TYPE>& get_dxInvs() const;
		PREFIX_VC_DLL
			FLOAT_TYPE get_dx(const size_t dimension) const;
		FLOAT_TYPE get_dxInv(const size_t dimension) const;
		PREFIX_VC_DLL
			const std::vector<std::vector<FLOAT_TYPE> >& get_vss() const;
		PREFIX_VC_DLL
			const std::vector<std::vector<FLOAT_TYPE> >& get_xss() const;
		PREFIX_VC_DLL
			const std::vector<FLOAT_TYPE>& get_vs(const size_t dimension) const;
		PREFIX_VC_DLL
			const std::vector<FLOAT_TYPE>& get_xs(const size_t dimension) const;
		PREFIX_VC_DLL
			void get_xs(
				beacls::UVec& x_uvec,
				const size_t dimension,
				const size_t start_index = 0,
				const size_t length = 0) const;
		PREFIX_VC_DLL
			void get_xss(
				std::vector<beacls::UVec>&  x_uvecs,
				const size_t start_index = 0,
				const size_t length = 0) const;
		const std::vector<FLOAT_TYPE>& get_axis() const;
		const std::vector<size_t>& get_shape() const;

		PREFIX_VC_DLL
			bool processGrid(const std::vector<FLOAT_TYPE> &data = std::vector<FLOAT_TYPE>());

		PREFIX_VC_DLL
			bool save_grid(
				const std::string &variable_name,
				beacls::MatFStream* fs,
				beacls::MatVariable* parent = NULL,
				const size_t cell_index = 0,
				const bool compress = true
			) const;
		PREFIX_VC_DLL
			bool load_grid(
				const std::string &variable_name,
				beacls::MatFStream* fs,
				beacls::MatVariable* parent = NULL,
				const size_t cell_index = 0);

		PREFIX_VC_DLL
			HJI_Grid* clone(const bool cloneAll = true) const;
	private:
		/** @overload
		Disable operator=
		*/
		HJI_Grid& operator=(const HJI_Grid& rhs);
		/** @overload
		Disable copy constructor
		*/
		HJI_Grid(const HJI_Grid& rhs);
	};
};
/*
@brief	Dump vector to text file.
@param	[in]	file_name	Filename.
@param	[in]	src_vector	Source vector.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool dump_vector(
	const std::string & file_name,
	const std::vector<FLOAT_TYPE> &src_vector);
/*
@brief	Save vector to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vector	Source vector.
@param	[in]	variable_name	Array name.
@param	[in]	src_grid	Grid structure for dimension information. If not specified,
deal with it as 1 dimension.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_double(
	const std::vector<double> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vector	Destination vector.
@param	[in]	variable_name	Array name,
which you want to load.
@param	[out]	dst_grid	Grid structure from file for dimension information.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_double(
	std::vector <double> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save vector to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vector	Source vector.
@param	[in]	variable_name	Array name.
@param	[in]	src_grid	Grid structure for dimension information. If not specified, deal with it as 1 dimension.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_float(
	const std::vector<float> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vector	Destination vector.
@param	[in]	variable_name	Array name,
which you want to load.
@param	[out]	dst_grid	Grid structure from file for dimension information.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_float(
	std::vector<float> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save vector to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vector	Source vector.
@param	[in]	variable_name	Array name.
@param	[in]	src_grid	Grid structure for dimension information. If not specified, deal with it as 1 dimension.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_size_t(
	const std::vector<size_t> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vector	Destination vector.
@param	[in]	variable_name	Array name,
which you want to load.
@param	[out]	dst_grid	Grid structure from file for dimension information.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_uint8_t(
	std::vector<uint8_t> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save vector to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vector	Source vector.
@param	[in]	variable_name	Array name.
@param	[in]	src_grid	Grid structure for dimension information. If not specified, deal with it as 1 dimension.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_uint8_t(
	const std::vector<uint8_t> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vector	Destination vector.
@param	[in]	variable_name	Array name,
which you want to load.
@param	[out]	dst_grid	Grid structure from file for dimension information.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_size_t(
	std::vector<size_t> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save vector to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vector	Source vector.
@param	[in]	variable_name	Array name.
@param	[in]	src_grid	Grid structure for dimension information. If not specified, deal with it as 1 dimension.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<typename T>
bool save_vector(
	const std::vector<T> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0,
	const bool compress = true);
template<>
inline
bool save_vector(
	const std::vector<double> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_double(src_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
template<>
inline
bool save_vector(
	const std::vector<float> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_float(src_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
template<>
inline
bool save_vector(
	const std::vector<size_t> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_size_t(src_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
template<>
inline
bool save_vector(
	const std::vector<uint8_t> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_uint8_t(src_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
/*
@brief	Load vector from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vector	Destination vector.
@param	[in]	variable_name	Array name,
which you want to load.
@param	[out]	dst_grid	Grid structure from file for dimension information.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<typename T>
bool load_vector(
	std::vector<T> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0);

template<>
inline
bool load_vector(
	std::vector<double> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_double(dst_vector, variable_name, Ns, quiet, fs, parent, cell_index);
}
template<>
inline
bool load_vector(
	std::vector<float> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_float(dst_vector, variable_name, Ns, quiet, fs, parent, cell_index);
}
template<>
inline
bool load_vector(
	std::vector<size_t> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_size_t(dst_vector, variable_name, Ns, quiet, fs, parent, cell_index);
}
template<>
inline
bool load_vector(
	std::vector<uint8_t> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_uint8_t(dst_vector, variable_name, Ns, quiet, fs, parent, cell_index);
}


/*
@brief	Save deque to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_deque	Source deque.
@param	[in]	variable_name	Array name.
@param	[in]	src_grid	Grid structure for dimension information. If not specified, deal with it as 1 dimension.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_deque_bool(
	const std::deque<bool> &src_deque,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load deque from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_deque	Destination deque.
@param	[in]	variable_name	Array name,
which you want to load.
@param	[out]	dst_grid	Grid structure from file for dimension information.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_deque_bool(
	std::deque<bool> &dst_deque,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save deque to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_deque	Source deque.
@param	[in]	variable_name	Array name.
@param	[in]	src_grid	Grid structure for dimension information. If not specified, deal with it as 1 dimension.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<typename T>
bool save_deque(
	const std::deque<T> &src_deque,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0,
	const bool compress = true);
template<>
inline
bool save_deque(
	const std::deque<bool> &src_deque,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_deque_bool(src_deque, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
/*
@brief	Load deque from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_deque	Destination deque.
@param	[in]	variable_name	Array name,
which you want to load.
@param	[out]	dst_grid	Grid structure from file for dimension information.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<typename T>
bool load_deque(
	std::deque<T> &dst_deque,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0);

template<>
inline
bool load_deque(
	std::deque<bool> &dst_deque,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_deque_bool(dst_deque, variable_name, Ns, quiet, fs, parent, cell_index);
}
/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_of_vectors_float(
	const std::vector<std::vector<float> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_of_vectors_float(
	std::vector<std::vector<float> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_of_vectors_double(
	const std::vector<std::vector<double> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_of_vectors_double(
	std::vector<std::vector<double> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_of_vectors_uint8_t(
	const std::vector<std::vector<uint8_t> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_of_vectors_uint8_t(
	std::vector<std::vector<uint8_t> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_of_vectors_int8_t(
	const std::vector<std::vector<int8_t> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_of_vectors_int8_t(
	std::vector<std::vector<int8_t> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);
/*
@brief	Load vector of deques from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_vector_of_deques_bool(
	std::vector<std::deque<bool> > &dst_deques,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);

/*
@brief	Save vector of deques to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_vector_of_deques_bool(
	const std::vector<std::deque<bool> > &src_deques,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);

template<typename T>
bool save_vector_of_vectors(
	const std::vector<std::vector<T> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0,
	const bool compress = true);
template<typename T>
bool load_vector_of_vectors(
	std::vector<std::vector<T> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0);


/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool save_vector_of_vectors(
	const std::vector<std::vector<float> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_float(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool load_vector_of_vectors(
	std::vector<std::vector<float> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_float(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool save_vector_of_vectors(
	const std::vector<std::vector<double> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_double(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool load_vector_of_vectors(
	std::vector<std::vector<double> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_double(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool save_vector_of_vectors(
	const std::vector<std::vector<uint8_t> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_uint8_t(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool load_vector_of_vectors(
	std::vector<std::vector<uint8_t> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_uint8_t(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
/*
@brief	Save vector of vectors to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of vectors.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool save_vector_of_vectors(
	const std::vector<std::vector<int8_t> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_int8_t(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
/*
@brief	Load vector of vectors from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of vectors.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool load_vector_of_vectors(
	std::vector<std::vector<int8_t> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_int8_t(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}


template<typename T>
bool save_vector_of_deques(
	const std::vector<std::deque<T> > &src_deques,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress);
template<typename T>
bool load_vector_of_deques(
	std::vector<std::deque<T> > &dst_deques,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index);


/*
@brief	Save vector of deques to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_vectors	Source vector of deques.
@param	[in]	variable_name	Vector name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool save_vector_of_deques(
	const std::vector<std::deque<bool> > &src_deques,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_deques_bool(src_deques, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
/*
@brief	Load vector of deques from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_vectors	Destination vector of deques.
@param	[in]	variable_name	Array name,
which you want to load.
@retval	true	Succeeded.
@retval	false	Failed
*/
template<>
inline
bool load_vector_of_deques(
	std::vector<std::deque<bool> > &dst_deques,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_deques_bool(dst_deques, variable_name, Ns, quiet, fs, parent, cell_index);
}

/*
@brief	Save value to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_value	Source value.
@param	[in]	value_name	Value name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_value(
	const float &src_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0,
	const bool compress = true);
/*
@brief	Load value from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_value	Destination value.
@param	[in]	value_name	Value name.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_value(
	float &dst_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0);
/*
@brief	Save value to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_value	Source value.
@param	[in]	value_name	Value name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_value(
	const double &src_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0,
	const bool compress = true);
/*
@brief	Load value from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_value	Destination value.
@param	[in]	value_name	Value name.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_value(
	double &dst_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0);

/*
@brief	Save value to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_value	Source value.
@param	[in]	value_name	Value name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_value(
	const size_t &src_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0,
	const bool compress = true);
/*
@brief	Load value from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_value	Destination value.
@param	[in]	value_name	Value name.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_value(
	size_t &dst_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0);

/*
@brief	Save value to MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[in]	src_value	Source value.
@param	[in]	value_name	Value name.
@param	[in]	append		Append to the existing file or create a new file.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool save_value(
	const bool &src_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0,
	const bool compress = true);
/*
@brief	Load value from MATLAB mat file (v7.3 format).
@param	[in]	file_name	Filename.
@param	[out]	dst_value	Destination value.
@param	[in]	value_name	Value name.
@arg	true	Append to the existing file.
@arg	false	Create a new file.
@retval	true	Succeeded.
@retval	false	Failed
*/
PREFIX_VC_DLL
bool load_value(
	bool &dst_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent = NULL,
	const size_t cell_index = 0);

#endif	/* __HJI_Grid_hpp__ */
