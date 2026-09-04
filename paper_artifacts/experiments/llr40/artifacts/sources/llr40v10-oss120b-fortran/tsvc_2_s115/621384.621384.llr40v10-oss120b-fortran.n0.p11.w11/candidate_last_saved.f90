module tsvc_2_s115_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C, name="tsvc_2_s115_fp64")
    real(c_double), intent(inout) :: a(0:*)
    real(c_double), intent(in)    :: aa(0:*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j
    ! Outer loop is sequential due to data dependencies; inner loop can be vectorized.
    do j = 0_c_int64_t, LEN_2D - 1_c_int64_t
      !$omp simd
      do i = j + 1_c_int64_t, LEN_2D - 1_c_int64_t
        a(i) = a(i) - aa(j * LEN_2D + i) * a(j)
      end do
    end do
  end subroutine tsvc_2_s115_fp64
end module tsvc_2_s115_mod
