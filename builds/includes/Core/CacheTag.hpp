#ifndef __CacheTag_hpp__
#define __CacheTag_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <cstdint>
#include <cstddef>
namespace levelset
{
	class CacheTag_impl;

	/*
	@brief CacheTag class
	*/
	class CacheTag {
	private:
		CacheTag_impl* pimpl;
	public:
		PREFIX_VC_DLL
			CacheTag(
			);
		PREFIX_VC_DLL
			~CacheTag();
		void set_tag(const FLOAT_TYPE new_t, const size_t new_bi, const size_t new_l);
		bool check_tag(const FLOAT_TYPE new_t, const size_t new_bi, const size_t new_l) const;
		bool check_tag(const size_t new_bi, const size_t new_l) const;
	private:
		/** @overload
		Disable operator=
		*/
		PREFIX_VC_DLL
			CacheTag& operator=(const CacheTag& rhs);
		/** @overload
		Disable copy constructor
		*/
		PREFIX_VC_DLL
			CacheTag(const CacheTag& rhs);
	};

}	// beacls
#endif	/* __CacheTag_hpp__ */

