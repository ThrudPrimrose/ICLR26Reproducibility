module wf_triangular_mod
  use iso_c_binding
  implicit none
contains
  subroutine wf_triangular_fp64(a, LEN_2D) bind(C, name="wf_triangular_fp64")
    ! Hand-written Fortran reference for wf_triangular kernel.
    ! Computes: a[i,j] += a[i-1,j] + a[i,j-1] for j >= i (row-major flattening).
    ! Arguments:
    !   a      - double-precision array, size LEN_2D*LEN_2D, row-major order.
    !   LEN_2D - dimension size (int64).
    real(c_double), dimension(*), intent(inout) :: a
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j, k
    integer(c_int64_t) :: i_start, i_end
    integer(c_int64_t) :: idx, idx_up, idx_left
    
    !$omp parallel private(i,j,k,i_start,i_end,idx,idx_up,idx_left)
    do k = 2_c_int64_t, 2_c_int64_t*LEN_2D - 2_c_int64_t
       i_start = max(1_c_int64_t, k - (LEN_2D - 1_c_int64_t))
       i_end   = min(LEN_2D - 1_c_int64_t, k/2_c_int64_t)
       !$omp do simd schedule(static)
       do i = i_start, i_end
          j = k - i
          idx      = i*LEN_2D + j
          idx_up   = (i-1_c_int64_t)*LEN_2D + j
          idx_left = i*LEN_2D + (j-1_c_int64_t)
          a(idx+1) = a(idx+1) + a(idx_up+1) + a(idx_left+1)
       end do
       !$omp end do simd
    end do
    !$omp end parallel
  end subroutine wf_triangular_fp64
end module wf_triangular_mod
