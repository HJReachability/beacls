#ifndef __ShapeHyperplaneByPoint_impl_hpp__
#define __ShapeHyperplaneByPoint_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
namespace levelset {

	class HJI_Grid;
	class ShapeHyperplaneByPoint_impl {
	private:
		std::vector<beacls::FloatVec > points;
		beacls::FloatVec positivePoint;
	public:
		ShapeHyperplaneByPoint_impl(
			const std::vector<beacls::FloatVec >& points,
			const  beacls::FloatVec& positivePoint);
		~ShapeHyperplaneByPoint_impl();
		bool execute(
			const HJI_Grid *grid, beacls::FloatVec& data) const;
		ShapeHyperplaneByPoint_impl* clone() const {
			return new ShapeHyperplaneByPoint_impl(*this);
		};
	private:
		/** @overload
		Disable operator=
		*/
		ShapeHyperplaneByPoint_impl& operator=(const ShapeHyperplaneByPoint_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		ShapeHyperplaneByPoint_impl(const ShapeHyperplaneByPoint_impl& rhs) :
			points(rhs.points),
			positivePoint(rhs.positivePoint)
		{}
	};
};
#endif	/* __ShapeHyperplaneByPoint_impl_hpp__ */

