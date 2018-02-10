/*
 * Copyright (c) 2017, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: Jaime F. Fisac   ( jfisac@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// DynSys subclass: relative dynamics between 5D Dubins car and 3D Dubins car.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef __P5D_Dubins_hpp__
#define __P5D_Dubins_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <typedef.hpp>
#include <cstddef>
#include <vector>
#include <iostream>
#include <cstring>
#include <utility>
using namespace std::rel_ops;
namespace helperOC {
    /*
        @note: Since plane is a "handle class", we can pass on
        handles/pointers to other plane objects
        e.g. a.platoon.leader = b (passes b by reference, does not create a copy)
        Also see constructor
    */
    class P5D_Dubins : public DynSys {
    public:
    protected:
        // Control bounds of this vehicle
        beacls::FloatVec aRange          // Linear acceleration
        FLOAT_TYPE alphaMax              // Angular acceleration

        // Vehicle speeds
        FLOAT_TYPE vOther

        // Control bounds of other vehicle (at origin)
        FLOAT_TYPE wMax     // Turn rate
    
        // Disturbance bounds
        beacls::FloatVec dMax     // 4D
    
    
        beacls::IntegerVec dims;    //!< Dimensions that are active
    public:
        /*
        @brief Constructor. Creates a plane object with a unique ID,
            state x, and reachable set information reachInfo
        Dynamics:
            \dot{x}_1 = v * cos(x_3) + d1
            \dot{x}_2 = v * sin(x_3) + d2
            \dot{x}_3 = u            + d3
                v \in [vrange(1), vrange(2)]
                u \in [-wMax, wMax]

        @param  [in]        x   state [xpos; ypos; theta]
        @param  [in]        uMax    maximum turn rate
        @param  [in]        dMax    maximum turn rate
        @param  [in]        va  Vehicle A Speeds
        @param  [in]        vd  Vehicle B Speeds
        @return a Plane object
        */
        PREFIX_VC_DLL
            P5D_Dubins(
                const beacls::FloatVec& x,
                const aRange = [-0.15, 0.15],
                const alphaMax = 3.0,
                const vOther = 0.1,
                const wMax = 2.0,
                const dMax = [0.02, 0.02, 0.2, 0.02],
                const beacls::IntegerVec& dims = beacls::IntegerVec{ 0,1,2,3,4 }
        );
        PREFIX_VC_DLL
            P5D_Dubins(
                beacls::MatFStream* fs,
                beacls::MatVariable* variable_ptr = NULL
            );
        PREFIX_VC_DLL
            virtual ~P5D_Dubins();
        PREFIX_VC_DLL
            virtual bool operator==(const P5D_Dubins& rhs) const;
        PREFIX_VC_DLL
            virtual bool operator==(const DynSys& rhs) const;
        PREFIX_VC_DLL
            virtual bool save(
                beacls::MatFStream* fs,
                beacls::MatVariable* variable_ptr = NULL
            );
        virtual P5D_Dubins* clone() const {
            return new P5D_Dubins(*this);
        }
        /*
        @brief Helper function for optimal inputs
        */
        PREFIX_VC_DLL
            bool Plane4D::optInput_i_cell_helper(
                beacls::FloatVec& uOpt_i,
                const std::vector<const FLOAT_TYPE* >& derivs,
                const beacls::IntegerVec& deriv_sizes,
                const helperOC::DynSys_UMode_Type uMode,
                const size_t src_target_dim_index,
                const beacls::FloatVec& uExtr_i
            ) const
        /*
        @brief Optimal control function
        */
        PREFIX_VC_DLL
            bool optCtrl(
                std::vector<beacls::FloatVec >& uOpts,
                const FLOAT_TYPE t,
                const std::vector<beacls::FloatVec::const_iterator >& y_ites,
                const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
                const beacls::IntegerVec& y_sizes,
                const beacls::IntegerVec& deriv_sizes,
                const helperOC::DynSys_UMode_Type uMode
            ) const;
        /*
        @brief Optimal disturbance function
        */
        PREFIX_VC_DLL
            bool optDstb(
                std::vector<beacls::FloatVec >& uOpts,
                const FLOAT_TYPE t,
                const std::vector<beacls::FloatVec::const_iterator >& y_ites,
                const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
                const beacls::IntegerVec& y_sizes,
                const beacls::IntegerVec& deriv_sizes,
                const helperOC::DynSys_DMode_Type dMode
            ) const;
        /*
        @brief Dynamics of the P5D_Dubins system
        */
        PREFIX_VC_DLL
            bool dynamics(
                std::vector<beacls::FloatVec >& dx,
                const FLOAT_TYPE t,
                const std::vector<beacls::FloatVec::const_iterator >& x_ites,
                const std::vector<beacls::FloatVec >& us,
                const std::vector<beacls::FloatVec >& ds,
                const beacls::IntegerVec& x_sizes,
                const size_t dst_target_dim
            ) const;


///////////////////////////////////////////////////////////////////////////////
//
//         Cuda functions below.  ---//---  Not implemented for now.
//
///////////////////////////////////////////////////////////////////////////////


// #if defined(USER_DEFINED_GPU_DYNSYS_FUNC) && 0
//         PREFIX_VC_DLL
//             bool optCtrl_cuda(
//                 std::vector<beacls::UVec>& u_uvecs,
//                 const FLOAT_TYPE t,
//                 const std::vector<beacls::UVec>& x_uvecs,
//                 const std::vector<beacls::UVec>& deriv_uvecs,
//                 const helperOC::DynSys_UMode_Type uMode
//             ) const;
//         /*
//         @brief Optimal disturbance function
//         */
//         PREFIX_VC_DLL
//             bool optDstb_cuda(
//                 std::vector<beacls::UVec>& d_uvecs,
//                 const FLOAT_TYPE t,
//                 const std::vector<beacls::UVec>& x_uvecs,
//                 const std::vector<beacls::UVec>& deriv_uvecs,
//                 const helperOC::DynSys_DMode_Type dMode
//             ) const;
//         /*
//         @brief Dynamics of the P5D_Dubins system
//         */
//         PREFIX_VC_DLL
//             bool dynamics_cuda(
//                 std::vector<beacls::UVec>& dx_uvecs,
//                 const FLOAT_TYPE t,
//                 const std::vector<beacls::UVec>& x_uvecs,
//                 const std::vector<beacls::UVec>& u_uvecs,
//                 const std::vector<beacls::UVec>& d_uvecs,
//                 const size_t dst_target_dim
//             ) const;
// #endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */

    private:
        /** @overload
        Disable operator=
        */
        P5D_Dubins& operator=(const P5D_Dubins& rhs);
        /** @overload
        Disable copy constructor
        */
        P5D_Dubins(const P5D_Dubins& rhs) :
            DynSys(rhs),
            uMax(rhs.uMax), //!< Control bounds
            dMax(rhs.dMax), //!< Control bounds
            va(rhs.va), //!< Vehicle Speeds
            vb(rhs.vb), //!< Vehicle Speeds
            dims(rhs.dims)  //!< Dimensions that are active
        {}
    };
};
#endif  /* __P5D_Dubins_hpp__ */
