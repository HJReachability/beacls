#include <Core/UVec.hpp>
#include <Core/CacheTag.hpp>
#include "CacheTag_impl.hpp"

#include <macro.hpp>

beacls::CacheTag::CacheTag(
) {
	pimpl = new CacheTag_impl();
}
beacls::CacheTag::~CacheTag() {
	if (pimpl) delete pimpl;
}
void beacls::CacheTag::set_tag(const FLOAT_TYPE new_t, const size_t new_bi, const size_t new_l) {
	if (pimpl) pimpl->set_tag(new_t, new_bi, new_l);
}
bool beacls::CacheTag::check_tag(const FLOAT_TYPE new_t, const size_t new_bi, const size_t new_l) const {
	if (pimpl) return pimpl->check_tag(new_t, new_bi, new_l);
	else return false;
}
