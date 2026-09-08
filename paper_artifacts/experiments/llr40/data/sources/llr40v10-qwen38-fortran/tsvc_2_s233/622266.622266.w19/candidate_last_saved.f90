module s233_mod
  use iso_c_binding, only: c_int64_t
  implicit none
contains
  subroutine s233_work(aa, bb, cc, n)
    double precision, intent(inout) :: aa(:), bb(:)
    double precision, intent(in)    :: cc(:)
    integer(c_int64_t), intent(in)  :: n

    integer(c_int64_t), parameter :: BW = 32
    integer(c_int64_t) :: i, j, ib, k, nk
    double precision :: s(BW)

    ! Part 1: per-column scan aa[j,i] = aa[j-1,i] + cc[j,i], i = 8..n-1.
!$omp parallel do schedule(static)
    do ib = 8, n-1, BW
      nk = min(BW, n - 8 - ib + 1)
      do k = 1, nk
        s(k) = aa(7*n + ib + k)
      end do
      if (nk == BW) then
        do j = 8, n-1
          do k = 1, BW
            s(k) = s(k) + cc(j*n + ib + k)
            aa(j*n + ib + k) = s(k)
          end do
        end do
      else
        do j = 8, n-1
          do k = 1, nk
            s(k) = s(k) + cc(j*n + ib + k)
            aa(j*n + ib + k) = s(k)
          end do
        end do
      end if
    end do
!$omp end parallel do

    ! Part 2: per-row scan bb[j,i] = bb[j,i-1] + cc[j,i], j = 8..n-1 (contiguous).
!$omp parallel do schedule(static)
    do j = 8, n-1
      s(1) = bb(j*n + 8)
      do i = 8, n-1
        s(1) = s(1) + cc(j*n + i + 1)
        bb(j*n + i + 1) = s(1)
      end do
    end do
!$omp end parallel do
  end subroutine
end module

subroutine tsvc_2_s233_fp64(aa, bb, cc, len_2d) bind(C, name='tsvc_2_s233_fp64')
  use s233_mod
  use iso_c_binding, only: c_ptr, c_int64_t
  implicit none
  type(c_ptr), intent(in) :: aa, bb, cc
  integer(c_int64_t), intent(in) :: len_2d

  double precision, pointer :: aa_f(:), bb_f(:), cc_f(:)
  double precision, target  :: dummy(1)
  integer(c_int64_t) :: n, n2

  n = len_2d
  if (n <= 8) return
  n2 = n*n
  call c_f_pointer(aa, aa_f, [n2], dummy)
  call c_f_pointer(bb, bb_f, [n2], dummy)
  call c_f_pointer(cc, cc_f, [n2], dummy)
  call s233_work(aa_f, bb_f, cc_f, n)
end subroutine
