#ifndef __CacheTag_impl_hpp__
#define __CacheTag_impl_hpp__

#include <cstdint>
#include <limits>
#include <Core/CacheTag.hpp>

namespace levelset
{

class CacheTag_impl {
public:
private:
	FLOAT_TYPE t;
	size_t begin_index;
	size_t length;
public:
	CacheTag_impl(
	) : t(std::numeric_limits<FLOAT_TYPE>::max()), begin_index(std::numeric_limits<size_t>::max()), length(std::numeric_limits<size_t>::max())
	{
	}
	~CacheTag_impl() {};
	void set_tag(const FLOAT_TYPE new_t, const size_t new_bi, const size_t new_l) {
		t = new_t;
		begin_index = new_bi;
		length = new_l;
	}
	bool check_tag(const FLOAT_TYPE new_t, const size_t new_bi, const size_t new_l) const {
		if ((t == new_t) && (begin_index == new_bi) && (length == new_l))return true;
		else return false;
	}
private:
	/** @overload
	Disable operator=
	*/
	CacheTag_impl& operator=(const CacheTag_impl& rhs);

	/** @overload
	copy constructor
	*/
	CacheTag_impl(const CacheTag_impl& rhs);
};


}	// beacls
#endif	/* __CacheTag_impl_hpp__ */

