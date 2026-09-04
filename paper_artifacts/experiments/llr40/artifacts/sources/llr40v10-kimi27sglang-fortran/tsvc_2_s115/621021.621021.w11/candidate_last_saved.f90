module tsvc_2_s115_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(c, name="tsvc_2_s115_fp64")
    real(c_double), intent(inout) :: a(LEN_2D)
    real(c_double), intent(in) :: aa(LEN_2D, LEN_2D)
    integer(c_int64_t), value, intent(in) :: LEN_2D
    integer(c_int64_t) :: j, i
    real(c_double) :: aj

    do j = 1, LEN_2D
        aj = a(j)
        !$omp simd
        do i = j + 1, LEN_2D
            a(i) = a(i) - aa(i, j) * aj
        end do
    end do
  end subroutine tsvc_2_s115_fp64
end module tsvc_2_s115_mod
