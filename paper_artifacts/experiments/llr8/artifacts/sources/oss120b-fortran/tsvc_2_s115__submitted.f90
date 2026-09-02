module tsvc_2_s115_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C, name="tsvc_2_s115_fp64")
    ! Arguments: a - inout vector, aa - input matrix, LEN_2D - size (int64)
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: aa(*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    real(c_double) :: a_j
    integer(c_int64_t) :: row_offset
    ! Loop over rows j, updating later elements i > j
    do j = 0, LEN_2D - 1
        ! Vectorizable inner loop: update a[i] using a[j] and aa[j,i]
      !$omp simd
      do i = j + 1, LEN_2D - 1
        a(i+1) = a(i+1) - aa(j*LEN_2D + i + 1) * a(j+1)
      end do
    end do
  end subroutine tsvc_2_s115_fp64
end module tsvc_2_s115_mod
