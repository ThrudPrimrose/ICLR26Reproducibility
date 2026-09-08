module tsvc_2_s115_mod
  use iso_c_binding, only: c_double, c_int64_t
  implicit none
contains
  subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C)
    ! Arguments: a - vector of length LEN_2D, aa - matrix LEN_2D x LEN_2D, LEN_2D - size
    integer(c_int64_t), value, intent(in) :: LEN_2D
    real(c_double), intent(inout) :: a(LEN_2D)
    real(c_double), intent(in)    :: aa(LEN_2D, LEN_2D)
    integer(c_int64_t) :: i, j

    !$omp parallel default(none) shared(a, aa, LEN_2D) private(j, i)
    do j = 1, LEN_2D
       !$omp do simd schedule(static)
       do i = j+1, LEN_2D
          a(i) = a(i) - aa(i, j) * a(j)
       end do
       !$omp end do simd
    end do
    !$omp end parallel
  end subroutine tsvc_2_s115_fp64
end module tsvc_2_s115_mod
