//=================================================================================================
/*!
//  \file blaze/math/typetraits/HasCosh.h
//  \brief Header file for the HasCosh type trait
//
//  Copyright (C) 2012-2020 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//  the terms of the New (Revised) BSD License. Redistribution and use in source and binary
//  forms, with or without modification, are permitted provided that the following conditions
//  are met:
//
//  1. Redistributions of source code must retain the above copyright notice, this list of
//     conditions and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice, this list
//     of conditions and the following disclaimer in the documentation and/or other materials
//     provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its contributors
//     may be used to endorse or promote products derived from this software without specific
//     prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_MATH_TYPETRAITS_HASCOSH_H_
#define _BLAZE_MATH_TYPETRAITS_HASCOSH_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <utility>
#include <blaze/math/typetraits/IsMatrix.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/util/EnableIf.h>
#include <blaze/util/IntegralConstant.h>
#include <blaze/util/typetraits/Void.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Auxiliary helper struct for the HasCosh type trait.
// \ingroup math_type_traits
*/
template< typename T, typename = void >
struct HasCoshHelper
   : public FalseType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCoshHelper type trait for types providing the cosh() operation.
// \ingroup math_type_traits
*/
template< typename T >
struct HasCoshHelper< T, Void_t< decltype( cosh( std::declval<T>() ) ) > >
   : public TrueType
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Availability of the cosh() operation for the given data types.
// \ingroup math_type_traits
//
// This type trait provides the information whether the cosh() operation exists for the given
// data type \a T (taking the cv-qualifiers into account). In case the operation is available,
// the \a value member constant is set to \a true, the nested type definition \a Type is
// \a TrueType, and the class derives from \a TrueType. Otherwise \a value is set to \a false,
// \a Type is \a FalseType, and the class derives from \a FalseType.

   \code
   struct NoCosh {};  // Definition of a type without the cosh() operation

   blaze::HasCosh< int >::value                  // Evaluates to 1
   blaze::HasCosh< DynamicVector<float> >::Type  // Results in TrueType
   blaze::HasCosh< DynamicMatrix<double> >       // Is derived from TrueType
   blaze::HasCosh< NoCosh >::value               // Evaluates to 0
   blaze::HasCosh< NoCosh >::Type                // Results in FalseType
   blaze::HasCosh< NoCosh >                      // Is derived from FalseType
   \endcode
*/
template< typename T, typename = void >
struct HasCosh
   : public HasCoshHelper<T>
{};
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCosh type trait for vectors.
// \ingroup math_type_traits
*/
template< typename T >
struct HasCosh< T, EnableIf_t< IsVector_v<T> > >
   : public HasCosh< typename T::ElementType >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*! \cond BLAZE_INTERNAL */
/*!\brief Specialization of the HasCosh type trait for matrices.
// \ingroup math_type_traits
*/
template< typename T >
struct HasCosh< T, EnableIf_t< IsMatrix_v<T> > >
   : public HasCosh< typename T::ElementType >
{};
/*! \endcond */
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Auxiliary variable template for the HasCosh type trait.
// \ingroup math_type_traits
//
// The HasCosh_v variable template provides a convenient shortcut to access the nested \a value
// of the HasCosh class template. For instance, given the type \a T the following two statements
// are identical:

   \code
   constexpr bool value1 = blaze::HasCosh<T>::value;
   constexpr bool value2 = blaze::HasCosh_v<T>;
   \endcode
*/
template< typename T >
constexpr bool HasCosh_v = HasCosh<T>::value;
//*************************************************************************************************

} // namespace blaze

#endif
