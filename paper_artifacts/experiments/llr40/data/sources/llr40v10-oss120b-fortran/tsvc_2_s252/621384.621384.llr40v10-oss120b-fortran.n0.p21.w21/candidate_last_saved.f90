module tsvc_2_s252_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s252_fp64(a, b, c, len_1d) bind(C, name="tsvc_2_s252_fp64")
    real(c_double), intent(out) :: a(*)
    real(c_double), intent(in) :: b(*), c(*)
    integer(c_int64_t), value :: len_1d
    integer(c_int64_t) :: i
    if (len_1d <= 0) return
    a(1) = b(1) * c(1)
    !$omp parallel do default(none) schedule(static) shared(a,b,c,len_1d) private(i)
    do i = 2, len_1d
       a(i) = b(i) * c(i) + b(i-1) * c(i-1)
    end do
    !$omp end parallel do
  end subroutine tsvc_2_s252_fp64
end module tsvc_2_s252_mod
