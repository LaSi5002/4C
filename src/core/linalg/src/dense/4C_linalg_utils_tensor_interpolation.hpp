// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_TENSOR_INTERPOLATION_HPP
#define FOUR_C_LINALG_UTILS_TENSOR_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /// enum class for the error types of the tensor interpolator
  enum class TensorInterpolationErrorType
  {
    NoErrors,             ///< no evaluation errors
    LinSolveFailQMatrix,  ///< the solution of the linear system of equations for the rotation
                          ///< matrix Q failed
    LinSolveFailRMatrix,  ///< the solution of the linear system of equations for the rotation
                          ///< matrix R failed
  };

  /// enum class for the interpolation type of the relative rotation matrices
  enum class RotationInterpolationType
  {
    RotationVector,  ///< interpolation of relative rotation vectors,
    Quaternion,      ///< interpolation of relative quaternions,
  };

  /// enum class for the interpolation type of the eigenvalue matrix
  enum class EigenvalInterpolationType
  {
    LOG,     ///< logarithmic weighted average,
    MLS,     ///< moving least squares,
    LOGMLS,  ///< logarithmic moving least squares,
  };

  /// make error message: error types to error message for the tensor interpolator
  inline std::string make_error_message(const TensorInterpolationErrorType err_type)
  {
    switch (err_type)
    {
      case TensorInterpolationErrorType::NoErrors:
        return "No Errors";
      case TensorInterpolationErrorType::LinSolveFailQMatrix:
        return "The solution of the linear system of equations for the rotation matrix Q "
               "(second-order tensor interpolation) has failed!";
      case TensorInterpolationErrorType::LinSolveFailRMatrix:
        return "The solution of the linear system of equations for the rotation matrix R "
               "(second-order tensor interpolation) has failed!";
      default:
        FOUR_C_THROW("to_string(Core::LinAlg::TensorInterpErrorType): you should not be here!");
    }
  }

  /*!
   * \class SecondOrderTensorInterpolator
   *
   * Interpolation of invertible second-order tensors (3x3), preserving tensor
   * characteristics.
   *
   * The class provides the capability to interpolate a second-order tensor \f$
   * \boldsymbol{T}_{\text{p}}
   * \f$ at the specified location  \f$ \boldsymbol{x}_\text{p} \f$, given a set of tensors \f$
   * \boldsymbol{T}_j \f$ (second-order, 3x3) at the spatial positions \f$ \boldsymbol{x}_j \f$.
   * The interpolation scheme, using a combined polar and spectral decomposition, preserves
   * several tensor characteristics, such as positive definiteness and monotonicity of
   * invariants. For further information on the interpolation scheme, refer to:
   * -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373
   *
   * @tparam loc_dim dimension of the location vectors \f$ \boldsymbol{x}_j \f$
   */

  template <unsigned int loc_dim>
  class SecondOrderTensorInterpolator
  {
   public:
    /*! @brief Constructor of the second-order tensor interpolator class
     *
     *  @param[in] order polynomial order (1:linear, 2: quadratic, ...) used for interpolating
     * the rotation vectors at the specified location
     *  @param[in] rot_interp_type interpolation algorithm used for
     *  the rotation matrices
     *  @param[in] eigenval_interp_type interpolation algorithm used for
     *  the eigenvalue matrices
     *  @param[in] interp_params interpolation parameters
     */
    SecondOrderTensorInterpolator(unsigned int order,
        const RotationInterpolationType rot_interp_type,
        const EigenvalInterpolationType eigenval_interp_type,
        const ScalarInterpolationParams& interp_params);

    /*! @brief Helper function to define the polynomial space
     *
     *  @param[in] order polynomial order (1:linear, 2: quadratic, ...) used for interpolating
     * the rotation vectors at the specified location
     *  @returns polynomial space(monomials) with desired polynomial order and dimensionality
     */
    Core::FE::PolynomialSpaceComplete<loc_dim, Core::FE::Polynomial> create_polynomial_space(
        unsigned int order);

    /*!
     * @brief Interpolate matrix (second-order 3x3 tensor) from a set of defined reference
     * matrices at specified locations.
     *
     * This method performs tensor interpolation based on a given set of tensors \f$
     * \boldsymbol{T}_j \f$ (second-order, 3x3) at the spatial positions/locations \f$
     * \boldsymbol{x}_j \f$. Concretely, the tensor is interpolated at the specified location \f$
     * \boldsymbol{x}_{\text{p}} \f$. Specifically, the R-LOG method from the paper below is
     * currently implemented (rotation vector interpolation + logarithmic weighted average method
     * for eigenvalues):
     * -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
     * Second-Order Tensors, Int J Number Methods Eng. 2024, 125, 10.1002/nme.7373
     * @param[in]  ref_matrices  reference 3x3 matrices \f$ \boldsymbol{T}_j \f$ used as basis for
     *                            interpolation
     * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
     * @param[in]  interp_loc location \f$ \boldsymbol{x}_{\text{p}} \f$ of the interpolated
     * tensor
     * @param[in, out] err_type  error type of the tensor interpolator
     *  (shall be TensorInterpErrorType::NoErrors if no errors occurred)
     * @returns interpolated 3x3 matrix
     */
    Core::LinAlg::Matrix<3, 3> get_interpolated_matrix(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc, TensorInterpolationErrorType& err_type);

    /*!
     * @brief Interpolate matrix (second-order 3x3 tensor) from a set of defined reference
     * matrices at specified locations.
     *
     * This method performs tensor interpolation based on a given set of tensors \f$
     * \boldsymbol{T}_j \f$ (second-order, 3x3) at the spatial positions/locations \f$
     * \boldsymbol{x}_j \f$. Concretely, the tensor is interpolated at the specified location \f$
     * \boldsymbol{x}_{\text{p}} \f$. Specifically, the R-LOG method from the paper below is
     * currently implemented (rotation vector interpolation + logarithmic weighted average method
     * for eigenvalues):
     * -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
     * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373
     * @param[in]  ref_matrices  reference 3x3 matrices \f$ \boldsymbol{T}_j \f$ used as basis for
     *                            interpolation
     * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
     * @param[in]  interp_loc location \f$ \boldsymbol{x}_{\text{p}} \f$ of the interpolated
     * tensor
     * @param[in, out] err_type  error type of the tensor interpolator
     *  (shall be TensorInterpErrorType::NoErrors if no errors occurred)
     * @returns interpolated 3x3 matrix
     */
    Core::LinAlg::Matrix<3, 3> get_interpolated_matrix(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<double>& ref_locs, const double interp_loc,
        TensorInterpolationErrorType& err_type);

    /*!
     * @name Get interpolation gradient, i.e. the derivative of the
     * interpolated matrix with respect to the interpolation location
     * vector / scalar.
     *
     * @note The derivative is computed numerically using a
     * perturbation approach with finite differences.
     *
     * @param[in]  ref_matrices  reference 3x3 matrices \f$ \boldsymbol{T}_j \f$ used as basis for
     *                            interpolation
     * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
     * @param[in]  interp_loc location \f$ \boldsymbol{x}_{\text{p}} \f$ of the interpolated
     * tensor
     * @param[in, out] err_type  error type of the tensor interpolator
     *  (shall be TensorInterpErrorType::NoErrors if no errors occurred)
     * @param[in]  perturbation_factor perturbation factor \f$
     * \epsilon_{\text{perturb}} \f$ to be used in the numerical
     * gradient determination (default: 1.0e-8).
     * @returns derivative of the interpolated matrix with respect to
     * the interpolation location vector. The result is a 9 x <dim>
     * matrix, where the second dimension corresponds to the dimension of
     * the location vector (1D, 2D, 3D).
     */
    //! @{
    /*!
     * @brief Standard method for interpolation locations with
     * variable dimensionality.
     */
    Core::LinAlg::Matrix<9, loc_dim> get_interpolation_gradient(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc, TensorInterpolationErrorType& err_type,
        const double perturbation_factor = 1.0e-8);

    /*!
     * @brief Specialized method for 1D interpolation locations.
     * @note Calls the standard method but is more easy to handle when
     * using 1D locations.
     */
    Core::LinAlg::Matrix<9, 1> get_interpolation_gradient(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<double>& ref_locs, const double interp_loc,
        TensorInterpolationErrorType& err_type, const double perturbation_factor = 1.0e-8);
    //! @}

   private:
    /// polynomial space used for the interpolation of rotation vectors depending
    /// on the desired order (created in constructor call)
    Core::FE::PolynomialSpaceComplete<loc_dim, Core::FE::Polynomial> polynomial_space_;

    /// rotation interpolation type
    const RotationInterpolationType rot_interp_type_;

    /// eigenvalue interpolation type
    const EigenvalInterpolationType eigenval_interp_type_;

    /// interpolation parameters
    const ScalarInterpolationParams interp_params_;
  };

  /*!
   * @brief Perform polar decomposition \f$ \boldsymbol{T} = \boldsymbol{R} \boldsymbol{U} \f$ of
   * the 3x3 invertible matrix
   * \f$ \boldsymbol{T} $
   *
   * This method performs Step 1 of the procedure described in:
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.5
   *
   *   Specifically, it splits a general tensor into its rotational and its stretch (symmetric,
   * positive definite) component. Moreover, the method calculates the eigenvalues, and it also
   * returns the spectral pairs of the tensor \f$ \boldsymbol{U} \f$, i.e., all 3 (eigenvalue,
   * eigenvector) eigenpairs. The spectral pairs are sorted in descending order of their
   * corresponding eigenvalues, while the diagonal eigenvalue matrix contains the lowest eigenvalue
   * in (0,0) and the highest in (2, 2).
   *
   * @param[in]  inp_matrix  input matrix \boldsymbol{T} to be decomposed
   * @param[out]  R_matrix  rotation matrix \boldsymbol{R}
   * @param[out]  U_matrix  stretch matrix \boldsymbol{U}
   * @param[out]  eigenval_matrix  eigenvalue matrix of the stretch matrix \boldsymbol{U}
   * @param[out]  spectral_pairs  vector of eigenpairs of the stretch matrix \boldsymbol{U}
   */
  void matrix_3x3_polar_decomposition(const Core::LinAlg::Matrix<3, 3>& inp_matrix,
      Core::LinAlg::Matrix<3, 3>& R_matrix, Core::LinAlg::Matrix<3, 3>& U_matrix,
      Core::LinAlg::Matrix<3, 3>& eigenval_matrix,
      std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& spectral_pairs);

  /*!
   * @brief Compute the symmetric, positive-definite material stretch $\boldsymbol{U}$ from the
   * invertible matrix \f$ \boldsymbol{T} = \boldsymbol{R}
   * \boldsymbol{U} \f$
   *
   * @param[in]  inp_matrix  input matrix \boldsymbol{T} to be decomposed
   * @return  material stretch \boldsymbol{U}
   */
  Core::LinAlg::Matrix<3, 3> matrix_3x3_material_stretch(
      const Core::LinAlg::Matrix<3, 3>& inp_matrix);

  /*!
   * @brief Compute the symmetric, positive-definite spatial stretch $\boldsymbol{v}$ from the
   * invertible matrix \f$ \boldsymbol{T} = \boldsymbol{v}
   * \boldsymbol{R} \f$
   *
   * @param[in]  inp_matrix  input matrix \boldsymbol{T} to be decomposed
   * @return  spatial stretch \boldsymbol{v}
   */
  Core::LinAlg::Matrix<3, 3> matrix_3x3_spatial_stretch(
      const Core::LinAlg::Matrix<3, 3>& inp_matrix);


  /*!
   * @brief Calculate the rotation vector from a given rotation matrix, using Spurrier's algorithm
   *
   *
   * For further information, refer to:
   *    -# Spurrier, Comment on "Singularity-Free Extraction of a Quaternion from a
   * Direction-Cosine
   * Matrix", Journal of Spacecraft and Rockets 1978, 15(4):255-255
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.2.2
   * @param[in]  rot_matrix  input rotation matrix
   * @returns  corresponding rotation vector
   */
  Core::LinAlg::Matrix<3, 1> calc_rot_vect_from_rot_matrix(
      const Core::LinAlg::Matrix<3, 3>& rot_matrix);


  /*!
   * @brief Calculate the rotation matrix from a given rotation vector, using the Rodrigues
   * formula
   *
   * For further information, refer to:
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.2.1
   * @param[in]  rot_vect  input rotation vector
   * @returns  corresponding rotation matrix
   */
  Core::LinAlg::Matrix<3, 3> calc_rot_matrix_from_rot_vect(
      const Core::LinAlg::Matrix<3, 1>& rot_vect);

  /*!
   * @brief Order the eigenpairs of a given matrix w.r.t. the eigenpairs of a reference
   * matrix to yield minimal rotations between corresponding eigenvectors (eigenvalues assumed
   * to already be sorted from highest to lowest in the eigenpairs)
   *
   * @note This ordering procedure is relevant in case of multiple eigenvalues, for which the
   * eigenpairs have to be ordered properly w.r.t. reference eigenpairs
   * For further information, refer to:
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for
   * Invertible Second-Order Tensors, Int J Number Methods Eng. 2024, 125, 10.1002/nme.7373,
   * Section 5.1
   *
   * @param[in]  ref_eigenpairs  eigenpairs of the reference matrix
   * @param[in|out]  eigenpairs  eigenpairs to be sorted w.r.t. reference matrix
   */
  void order_eigenpairs_wrt_reference(
      const std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& ref_eigenpairs,
      std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& eigenpairs);

  /*!
   * @brief Align the eigenpairs of the base matrix (in the case of
   * tensor interpolation: nearest to the interpolation point) to
   * account for multiple eigenvalues
   *
   *  The eigenpairs of the base matrix are reordered in case of multiple eigenvalues to yield
   *  minimal rotations w.r.t. the eigenpairs of the other matrices.
   *  Theoretically, some matrices will be favored in this reordering process, since there are
   *  max. 6 possible ways to reorder the eigenvectors of the base matrix (for a triple
   * eigenvalue). The following criteria determine the reordering result (priority: 1-> highest):
   *  1. Distance of the location point (the matrix whose location lies nearest to the base
   * matrix is favored)
   *  2. Highest eigenvalue (the matrix with the overall highest eigenvalue is favored in the
   * reordering process)
   *
   * @param[in|out]  spectral_pairs  all spectral pairs (eigenvalue, eigenvector) of all
   *                                 available matrices used for interpolation
   * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
   * @param[in]  base_ind  index of the base matrix within spectral_pairs
   */
  template <unsigned int loc_dim>
  void align_eigenpairs_of_base_matrix(
      std::vector<std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>>& spectral_pairs,
      const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs, const unsigned int& base_ind);

}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif
