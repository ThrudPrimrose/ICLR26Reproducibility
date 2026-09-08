module tsvc_2_s1232_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s1232_fp64(a, b, c, len_2d, vlen) bind(C, name="tsvc_2_s1232_fp64")
    integer(c_int64_t), value, intent(in) :: len_2d
    integer(c_int64_t), value, intent(in) :: vlen
    real(c_double), intent(inout) :: a(len_2d, len_2d)
    real(c_double), intent(in) :: b(len_2d, len_2d)
    real(c_double), intent(in) :: c(len_2d, len_2d)
    integer(c_int64_t) :: i, j, max_j
    !$omp parallel do default(none) schedule(static) shared(a,b,c,len_2d,vlen) private(i,j,max_j)
    do i = 1_c_int64_t, len_2d
      max_j = (i - 1_c_int64_t) / vlen + 1_c_int64_t
      if (max_j > len_2d) max_j = len_2d
      !$omp simd
      do j = 1_c_int64_t, max_j
        a(j,i) = b(j,i) + c(j,i)
      end do
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s1232_fp64
end module tsvc_2_s1232_mod
