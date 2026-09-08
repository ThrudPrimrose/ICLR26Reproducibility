module wfmod
  use, intrinsic :: iso_c_binding
  implicit none
contains
  subroutine wf_triangular_fp64(a_ptr, LEN_2D) bind(C, name="wf_triangular_fp64")
    implicit none
    type(c_ptr), value :: a_ptr
    integer(kind=c_int64_t), value :: LEN_2D

    real(kind=c_double), pointer :: a(:)
    integer(kind=c_int64_t) :: ii, jj, n

    n = LEN_2D
    call c_f_pointer(a_ptr, a, [n * n])
    do ii = 1, n - 1            ! 0-based row
      do jj = ii, n - 1         ! 0-based col
        a(ii * n + jj + 1) = a(ii * n + jj + 1) + a((ii - 1) * n + jj + 1) + a(ii * n + jj)
      end do
    end do
  end subroutine wf_triangular_fp64
end module wfmod
