module tsvc_2_s115_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C, name="tsvc_2_s115_fp64")
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: aa(*)
    integer(c_int64_t), value :: LEN_2D
    integer(c_int64_t) :: i, j, offset
    real(c_double) :: a_j
    !$omp parallel private(j,i)
    do j = 1, LEN_2D - 1
      !$omp single
      a_j = a(j)
      offset = (j - 1) * LEN_2D
      !$omp end single
      !$omp do simd schedule(static)
      do i = j + 1, LEN_2D
        a(i) = a(i) - aa(offset + i) * a_j
      end do
      !$omp end do simd
    end do
    !$omp end parallel
  end subroutine tsvc_2_s115_fp64
end module tsvc_2_s115_mod
